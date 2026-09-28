"""Time one steered prefill and a short greedy decode on the selected model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

from hybrid_steering import Runner, gdn_layers, load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
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
    path = Path(__file__).resolve().parents[1] / "forgetting" / "questions.py"
    spec = importlib.util.spec_from_file_location("forgetting_questions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    texts = [row["question"] for row in module.simple_questions(args.batch_size, seed=42)]
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
