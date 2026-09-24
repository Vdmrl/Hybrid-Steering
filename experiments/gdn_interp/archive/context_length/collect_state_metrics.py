"""Collect generic GDN state-matrix metrics (norm, rank, cosine, ...) along a
steered, teacher-forced replay, for any steering mode.

Unlike collect_propagation.py's rank-one trajectory (which only tracks the
decay of a single S_0 injection and can't represent 'repeated'/'periodic'/
'prompt_end' re-injection events), this reads the actual recurrent state
directly via gdn_interp.dynamics.matrix_metrics -- valid for any mode, at the
cost of not decomposing *why* the state looks the way it does.

Replay uses exactly the same forward-call granularity as real generation, so
that steer_states' periodic/repeated call counting lines up with what the
saved responses were actually generated under: the prompt is a single bulk
forward call (matching prefill), and the response is replayed one token at a
time (matching decode steps). Chunking the prompt into multiple calls would
add spurious periodic/repeated injections that never happened during
generation, so prompt-phase state is only sampled at its start and end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gdn_interp import gdn_layers, load_qwen, steer_states, steering_deltas
from gdn_interp.dynamics import matrix_metrics

LENGTHS = (0, 128, 256, 512, 1024, 2048, 4096, 8192)
SCALARS = (
    "frobenius_norm",
    "sigma_1",
    "effective_rank",
    "stable_rank",
    "state_cosine",
)


def load_responses(run_dir: Path, scale: float, lengths: tuple[int, ...], limit: int):
    contexts = {
        (row["example_id"], row["target_tokens"]): row
        for row in map(json.loads, (run_dir / "contexts.jsonl").read_text().splitlines())
    }
    responses = [
        row
        for row in map(json.loads, (run_dir / "responses.jsonl").read_text().splitlines())
        if row["scale"] == scale and row["target_tokens"] in lengths
    ]
    by_length: dict[int, list[dict]] = {}
    for row in sorted(responses, key=lambda r: (r["target_tokens"], r["example_id"])):
        group = by_length.setdefault(row["target_tokens"], [])
        if len(group) < limit:
            group.append(row)
    return contexts, by_length


def snapshot(cache, decoder_layers: list[int]) -> torch.Tensor:
    """[layer, head, key, value] recurrent state, all heads."""
    return torch.stack(
        [cache.layers[layer].recurrent_states[0][0] for layer in decoder_layers]
    ).detach()


def replay_one(
    model,
    tokenizer,
    row: dict,
    prompt_ids: list[int],
    deltas: dict[int, torch.Tensor],
    mode: str,
    period: int,
    decoder_layers: list[int],
    device: torch.device,
    svd_driver: str | None,
) -> dict[str, torch.Tensor]:
    """Replay one example; return per-response-position metric tensors [pos, layer, head]."""
    response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    positions: list[dict[str, torch.Tensor]] = []
    prompt_boundary: dict[str, torch.Tensor] | None = None
    with torch.inference_mode(), steer_states(
        model, deltas, mode=mode, period=period, batch_size=1
    ) as cache:
        model.model(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            past_key_values=cache,
            use_cache=True,
        )
        previous_states = snapshot(cache, decoder_layers)
        full_metrics, previous_vh = matrix_metrics(previous_states, svd_driver=svd_driver)
        prompt_boundary = {key: full_metrics[key] for key in SCALARS}
        for token_id in response_ids:
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            model.model(
                input_ids=token,
                attention_mask=torch.ones_like(token),
                past_key_values=cache,
                use_cache=True,
            )
            states = snapshot(cache, decoder_layers)
            metrics, previous_vh = matrix_metrics(
                states, previous_states, previous_vh, svd_driver=svd_driver
            )
            previous_states = states
            positions.append(metrics)
    stacked = {
        key: torch.stack([position[key] for position in positions])
        for key in SCALARS
        if positions
    }
    return prompt_boundary, stacked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect state-matrix metrics from a steered, teacher-forced replay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=14.0)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--limit", type=int, default=20, help="examples per length")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--svd-driver", choices=("gesvdj", "gesvda", "gesvd"), default="gesvdj"
    )
    args = parser.parse_args()

    direction = torch.load(args.run_dir / "direction.pt", weights_only=True)
    generation = json.loads((args.run_dir / "generation.json").read_text())
    normalization = generation.get("normalization", "rms")
    mode = generation.get("mode", "initial")
    period = generation.get("period") or 1
    deltas = steering_deltas(
        direction["direction"],
        range(len(direction["gdn_layers"])),
        args.scale,
        direction["state_rms"] if normalization == "rms" else None,
    )

    contexts, by_length = load_responses(
        args.run_dir, args.scale, tuple(args.lengths), args.limit
    )

    model, tokenizer = load_qwen(
        direction["model"], device_map={"": 0}, dtype=getattr(torch, args.dtype)
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    found = gdn_layers(model)
    decoder_layers = [module.layer_idx for _, module in found]
    heads = list(range(found[0][1].num_v_heads))

    output_dir = args.run_dir / "intermediate" / "state_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    for length, rows in sorted(by_length.items()):
        prompt_totals = None
        response_totals: dict[str, torch.Tensor] = {}
        response_counts: dict[str, torch.Tensor] = {}
        n = 0
        for index, row in enumerate(rows, 1):
            prompt_ids = contexts[(row["example_id"], row["target_tokens"])][
                "prompt_input_ids"
            ]
            prompt_metrics, response_metrics = replay_one(
                model,
                tokenizer,
                row,
                prompt_ids,
                deltas,
                mode,
                period,
                decoder_layers,
                device,
                args.svd_driver,
            )
            prompt_totals = (
                {key: value.double().cpu() for key, value in prompt_metrics.items()}
                if prompt_totals is None
                else {
                    key: prompt_totals[key] + prompt_metrics[key].double().cpu()
                    for key in prompt_totals
                }
            )
            for key, value in response_metrics.items():
                value = value.double().cpu()
                if key not in response_totals:
                    response_totals[key] = torch.zeros_like(value)
                    response_counts[key] = torch.zeros(value.shape[0], dtype=torch.int32)
                width = value.shape[0]
                if response_totals[key].shape[0] < width:
                    pad = width - response_totals[key].shape[0]
                    response_totals[key] = torch.nn.functional.pad(
                        response_totals[key], (0, 0, 0, 0, 0, pad)
                    )
                    response_counts[key] = torch.nn.functional.pad(
                        response_counts[key], (0, pad)
                    )
                response_totals[key][:width] += value
                response_counts[key][:width] += 1
            n += 1
            print(f"length={length} mode={mode} {index}/{len(rows)}", flush=True)
        torch.save(
            {
                "model": direction["model"],
                "mode": mode,
                "period": period,
                "scale": args.scale,
                "length": length,
                "examples": n,
                "decoder_layers": decoder_layers,
                "heads": heads,
                "metric_names": SCALARS,
                "prompt_boundary_mean": {
                    key: (value / n).float() for key, value in prompt_totals.items()
                }
                if prompt_totals
                else {},
                "response_mean": {
                    key: (response_totals[key] / response_counts[key][:, None, None]).float()
                    for key in response_totals
                },
                "response_count": {key: response_counts[key] for key in response_counts},
            },
            output_dir / f"length-{length:04d}.pt",
        )
        print(f"wrote {output_dir / f'length-{length:04d}.pt'}")


if __name__ == "__main__":
    main()
