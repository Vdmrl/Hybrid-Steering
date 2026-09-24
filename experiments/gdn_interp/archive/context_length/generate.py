"""Resume steered generation in equal-length batches (no padding).

Steering timing is controlled by --mode:
  initial     S_0 = delta once before prefill, never touched again (the
              original position-0 experiment).
  repeated    S_0 = delta, then re-add delta after every forward call, i.e.
              every generated token (period=1).
  periodic    S_0 = delta, then re-add delta every --period forward calls
              (e.g. every 32/64/128 tokens).
  prompt_end  no S_0 preset; add delta exactly once, right after the prompt
              is fully processed, before the first generated token.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from gdn_interp import load_qwen, steer_states, steering_deltas

DEFAULT_STEERING_SCALES = (1.25, 1.5)
DEFAULT_MAX_NEW_TOKENS = 256
STEERING_MODES = ("initial", "repeated", "periodic", "prompt_end")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--scales", type=float, nargs="+", default=DEFAULT_STEERING_SCALES)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--batch-tokens",
        type=int,
        default=262144,
        help="cap prompt tokens per batch; protects long-context memory",
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lengths", type=int, nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--normalization", choices=("raw", "rms"), default="raw")
    p.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="pinned to match the fp16 numerics the run's artifacts were generated under",
    )
    p.add_argument("--mode", choices=STEERING_MODES, default="initial")
    p.add_argument(
        "--period",
        type=int,
        default=1,
        help="re-add delta every N forward calls; only used with --mode periodic",
    )
    args = p.parse_args()
    if args.mode == "periodic" and args.period < 1:
        raise ValueError("--period must be >= 1")
    contexts = args.run_dir / "contexts.jsonl"
    direction = args.run_dir / "direction.pt"
    output_path = args.run_dir / "responses.jsonl"
    generation_path = args.run_dir / "generation.json"
    if output_path.exists() and output_path.stat().st_size and generation_path.exists():
        previous = json.loads(generation_path.read_text()).get("normalization", "rms")
        if previous != args.normalization:
            raise ValueError(
                f"refusing to mix {previous} and {args.normalization} steering in one run"
            )
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    period = args.period if args.mode == "periodic" else None
    generation_path.write_text(
        json.dumps(
            {
                "scales": args.scales,
                "normalization": args.normalization,
                "batch_size": args.batch_size,
                "batch_tokens": args.batch_tokens,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "lengths": args.lengths,
                "limit": args.limit,
                "include_baseline": not args.skip_base,
                "mode": args.mode,
                "period": period,
            },
            indent=2,
        )
        + "\n"
    )
    artifact = torch.load(direction, weights_only=True)
    model, tokenizer = load_qwen(artifact["model"], dtype=getattr(torch, args.dtype))
    model.eval()
    device = model.get_input_embeddings().weight.device
    rows = [json.loads(x) for x in contexts.read_text().splitlines()]
    if args.lengths:
        rows = [row for row in rows if row["target_tokens"] in args.lengths]
    if args.limit:
        rows = rows[: args.limit]

    def done_key(example_id, target_tokens, scale, mode, period):
        # baseline (scale=0) is unsteered, so it's shared across every mode
        if scale == 0:
            return (example_id, target_tokens, scale)
        return (example_id, target_tokens, scale, mode, period)

    done = set()
    if output_path.exists():
        done = {
            done_key(
                x["example_id"],
                x["target_tokens"],
                x["scale"],
                x.get("steering_mode", "initial"),
                x.get("period"),
            )
            for x in map(json.loads, output_path.read_text().splitlines())
        }
    groups = defaultdict(list)
    for row in rows:
        groups[row["input_tokens"]].append(row)
    deltas = {
        s: steering_deltas(
            artifact["direction"],
            range(len(artifact["gdn_layers"])),
            s,
            artifact["state_rms"] if args.normalization == "rms" else None,
        )
        for s in args.scales
    }
    with output_path.open("a") as out, torch.inference_mode():
        for length, group in sorted(groups.items()):
            batch_size = min(args.batch_size, max(1, args.batch_tokens // length))
            for scale in (*args.scales,) if args.skip_base else (0.0, *args.scales):
                pending = [
                    r
                    for r in group
                    if done_key(r["example_id"], r["target_tokens"], scale, args.mode, period)
                    not in done
                ]
                for start in range(0, len(pending), batch_size):
                    batch = pending[start : start + batch_size]
                    input_ids = torch.tensor(
                        [row["prompt_input_ids"] for row in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": torch.ones_like(input_ids),
                    }
                    if scale == 0:
                        output = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                        )
                    else:
                        with steer_states(
                            model,
                            deltas[scale],
                            mode=args.mode,
                            period=args.period,
                            batch_size=len(batch),
                        ) as cache:
                            output = model.generate(
                                **inputs,
                                past_key_values=cache,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                            )
                    width = input_ids.shape[1]
                    for row, ids in zip(batch, output, strict=True):
                        metadata = {
                            key: value
                            for key, value in row.items()
                            if key
                            not in {"filler", "filler_input_ids", "prompt_input_ids"}
                        }
                        out.write(
                            json.dumps(
                                {
                                    **metadata,
                                    "scale": scale,
                                    "normalization": args.normalization,
                                    "steering_mode": args.mode if scale else None,
                                    "period": period if scale else None,
                                    "response": tokenizer.decode(
                                        ids[width:], skip_special_tokens=True
                                    ),
                                }
                            )
                            + "\n"
                        )
                    out.flush()
                    print(
                        f"length={length} scale={scale} mode={args.mode} "
                        f"{start + len(batch)}/{len(pending)}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
