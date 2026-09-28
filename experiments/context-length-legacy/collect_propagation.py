"""Collect rank-one GDN steering propagation along saved generations."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F

from gdn_interp import gdn_layers, load_qwen, steer_states, steering_deltas
from gdn_interp.dynamics import rank_one_trajectory

LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)
METRICS = ("sigma_t", "cosine_u_q", "norm_product")
SCALE = 14.0


def aggregate(raw_dir: Path, output: Path) -> None:
    totals: dict[int, torch.Tensor] = {}
    counts: dict[int, torch.Tensor] = {}
    metadata = None
    for path in sorted(raw_dir.glob("metrics-*.pt")):
        shard = torch.load(path, weights_only=True)
        metadata = shard
        for index, total_tokens in enumerate(shard["total_tokens"].tolist()):
            length = int(shard["target_tokens"][index])
            values = shard["metrics"][index, :total_tokens].double()
            if length not in totals:
                totals[length] = torch.zeros_like(values)
                counts[length] = torch.zeros(total_tokens, dtype=torch.int32)
            elif total_tokens > totals[length].shape[0]:
                padding = total_tokens - totals[length].shape[0]
                totals[length] = F.pad(totals[length], (0, 0, 0, 0, 0, 0, 0, padding))
                counts[length] = F.pad(counts[length], (0, padding))
            totals[length][:total_tokens] += values
            counts[length][:total_tokens] += 1
    if metadata is None:
        raise ValueError(f"no metric shards found in {raw_dir}")
    torch.save(
        {
            "metric_names": METRICS,
            "scale": SCALE,
            "gdn_layers": metadata["gdn_layers"],
            "decoder_layers": metadata["decoder_layers"],
            "heads": metadata["heads"],
            "by_length": {
                length: {
                    "mean": (totals[length] / counts[length][:, None, None, None]).float(),
                    "count": counts[length],
                    "prompt_tokens": length,
                }
                for length in sorted(totals)
            },
        },
        output,
    )


def main() -> None:
    global LENGTHS, SCALE
    parser = argparse.ArgumentParser(
        description="Collect propagation metrics from saved steered generations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-tokens", type=int, default=65536)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="pinned to match the fp16 numerics the run's artifacts were generated under",
    )
    args = parser.parse_args()
    LENGTHS = tuple(args.lengths)
    SCALE = args.scale

    direction = torch.load(args.run_dir / "direction.pt", weights_only=True)
    generation = json.loads((args.run_dir / "generation.json").read_text())
    normalization = generation.get("normalization", "rms")
    steering_mode = generation.get("mode", "initial")
    if steering_mode != "initial":
        raise ValueError(
            f"rank-one propagation replay only supports mode='initial' runs "
            f"(got {steering_mode!r}): the analytic sigma_t/cosine trajectory "
            "tracks decay of a single S_0 injection and does not model "
            "re-injected delta events from 'repeated', 'periodic', or "
            "'prompt_end' generation. Use state metrics and custom-judge "
            "results for those modes instead."
        )
    deltas = steering_deltas(
        direction["direction"],
        range(len(direction["gdn_layers"])),
        SCALE,
        direction["state_rms"] if normalization == "rms" else None,
    )
    contexts = {
        (row["example_id"], row["target_tokens"]): row
        for row in map(
            json.loads, (args.run_dir / "contexts.jsonl").read_text().splitlines()
        )
    }
    responses = [
        row
        for row in map(
            json.loads, (args.run_dir / "responses.jsonl").read_text().splitlines()
        )
        if row["target_tokens"] in LENGTHS and row["scale"] == SCALE
    ]
    responses.sort(key=lambda row: (row["target_tokens"], row["example_id"]))
    if args.limit:
        responses = responses[: args.limit]
    if not responses:
        raise ValueError("no matching scale-1.25 responses")

    output_dir = args.run_dir / "intermediate" / "propagation"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_qwen(direction["model"], device_map={"": 0}, dtype=getattr(torch, args.dtype))
    model.eval()
    device = model.get_input_embeddings().weight.device
    found = gdn_layers(model)
    modules = [module for _, module in found]
    decoder_layers = [module.layer_idx for module in modules]
    heads = list(range(modules[0].num_v_heads))

    factors = {}
    for ordinal, delta in deltas.items():
        u, sigma, _ = torch.linalg.svd(
            delta.to(device=device, dtype=torch.float32), full_matrices=False
        )
        factors[modules[ordinal]] = (u[..., 0], sigma[..., 0])

    conv_outputs = {}
    captured = {}

    def capture_conv(module, _args, output):
        conv_outputs[module] = output

    def capture_metrics(module, args_, kwargs, _output):
        hidden = kwargs.get("hidden_states", args_[0] if args_ else None)
        if hidden is None:
            raise RuntimeError("GDN hook could not find hidden_states")
        batch_size, sequence_length, _ = hidden.shape
        raw = conv_outputs.pop(module.conv1d)
        convolved = F.silu(
            raw[:, :, : sequence_length + module.conv_kernel_size]
        )[:, :, -sequence_length:]
        query, key, _ = torch.split(
            convolved.transpose(1, 2),
            (module.key_dim, module.key_dim, module.value_dim),
            dim=-1,
        )
        query = query.reshape(
            batch_size, sequence_length, -1, module.head_k_dim
        )
        key = key.reshape(batch_size, sequence_length, -1, module.head_k_dim)
        repeats = module.num_v_heads // module.num_k_heads
        if repeats > 1:
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        query = query * torch.rsqrt(
            query.square().sum(-1, keepdim=True) + 1e-6
        )
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        query = query / math.sqrt(module.head_k_dim)
        beta = module.in_proj_b(hidden).sigmoid()
        log_alpha = -module.A_log.float().exp() * F.softplus(
            module.in_proj_a(hidden).float() + module.dt_bias
        )
        initial_u, initial_sigma = factors[module]
        captured[module] = rank_one_trajectory(
            initial_u,
            initial_sigma,
            query,
            key,
            log_alpha.exp(),
            beta,
        ).cpu()

    with ExitStack() as stack:
        for module in modules:
            stack.callback(
                module.conv1d.register_forward_hook(capture_conv).remove
            )
            stack.callback(
                module.register_forward_hook(
                    capture_metrics, with_kwargs=True
                ).remove
            )
        for length in LENGTHS:
            group = [
                row
                for row in responses
                if row["target_tokens"] == length
                and not (
                    raw_dir
                    / f"metrics-l{length:04d}-e{row['example_id']:04d}.pt"
                ).exists()
            ]
            batch_size = min(
                args.batch_size,
                max(1, args.batch_tokens // (length + 256)),
            )
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                sequences = []
                prompt_tokens = []
                for row in batch:
                    prompt = contexts[(row["example_id"], length)]["prompt_input_ids"]
                    answer = tokenizer.encode(row["response"], add_special_tokens=False)
                    sequences.append(prompt + answer)
                    prompt_tokens.append(len(prompt))
                total_tokens = torch.tensor([len(ids) for ids in sequences])
                width = int(total_tokens.max())
                input_ids = torch.full(
                    (len(batch), width),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for index, ids in enumerate(sequences):
                    input_ids[index, : len(ids)] = torch.tensor(ids, device=device)
                    attention_mask[index, : len(ids)] = 1
                captured.clear()
                with torch.inference_mode(), steer_states(
                    model, deltas, batch_size=len(batch)
                ) as cache:
                    model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        use_cache=True,
                    )
                metrics = torch.stack([captured[module] for module in modules], dim=2)
                for index, row in enumerate(batch):
                    output = (
                        raw_dir
                        / f"metrics-l{length:04d}-e{row['example_id']:04d}.pt"
                    )
                    torch.save(
                        {
                            "example_ids": torch.tensor([row["example_id"]]),
                            "target_tokens": torch.tensor([row["target_tokens"]]),
                            "prompt_tokens": torch.tensor([prompt_tokens[index]]),
                            "total_tokens": total_tokens[index : index + 1],
                            "scale": SCALE,
                            "metric_names": METRICS,
                            "gdn_layers": [name for name, _ in found],
                            "decoder_layers": decoder_layers,
                            "heads": heads,
                            "metrics": metrics[
                                index : index + 1, : int(total_tokens[index])
                            ].clone(),
                        },
                        output,
                    )
                print(
                    f"length={length} {start + len(batch)}/{len(group)}",
                    flush=True,
                )

    aggregate(raw_dir, output_dir / "averages.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": direction["model"],
                "scale": SCALE,
                "normalization": normalization,
                "lengths": LENGTHS,
                "metrics": METRICS,
                "responses": len(responses),
                "rank_one_rule": "sigma_t = sigma_0 * ||u_t||",
                "trajectory": "steered teacher-forced prompt plus saved response",
                "query": "kernel L2-normalized query divided by sqrt(key dimension)",
                "key": "kernel L2-normalized key",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output_dir / 'averages.pt'}")


if __name__ == "__main__":
    main()
