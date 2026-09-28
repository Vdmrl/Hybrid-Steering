"""Time one steered prefill and a short greedy decode on the selected model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hybrid_steering import Runner, gdn_layers, load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "tiny"
        args.batch_size = 2
        args.max_new_tokens = 2
        args.repeats = 1
    model, tokenizer = load_runtime(args.model)
    layers = gdn_layers(model)
    device = next(model.parameters()).device
    deltas = {
        layer: torch.ones(
            model.model.layers[layer].linear_attn.num_v_heads,
            model.model.layers[layer].linear_attn.head_k_dim,
            model.model.layers[layer].linear_attn.head_v_dim,
            device=device,
        )
        for layer in layers
    }
    runner = Runner(model, tokenizer, layers, deltas, normalize=False)
    texts = [f"Prompt number {index}." for index in range(args.batch_size)]
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        tokens = runner.generate(
            texts, scale=0.1, prompt_position=-1, max_new_tokens=args.max_new_tokens
        )
        samples.append((time.perf_counter() - start) * 1e3)
    if not torch.isfinite(tokens).all():
        raise SystemExit("benchmark generated non-finite token ids")
    payload = {"model": args.model, "milliseconds": samples, "tokens": int(tokens.numel())}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
