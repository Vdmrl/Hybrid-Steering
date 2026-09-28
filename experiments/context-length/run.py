"""Context-length steering modes on one direction.

``initial`` writes the direction before the first real token.
``prompt-end`` writes it at the last prompt token.
``repeated`` and ``periodic`` refresh it during generation.
The initial state is zero, so these modes use ``normalize=False``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_steering import Runner, gdn_layers, load_runtime

MODES = ("initial", "prompt-end", "repeated", "periodic")
PROMPTS = (
    "Short question about rain.",
    "A slightly longer question about missing the evening bus home.",
)


def generate(
    runner: Runner, mode: str, scale: float, period: int, max_new_tokens: int
) -> torch.Tensor:
    if mode == "initial":
        return runner.generate(
            list(PROMPTS), scale=scale, prompt_position=0, max_new_tokens=max_new_tokens
        )
    if mode == "prompt-end":
        return runner.generate(
            list(PROMPTS), scale=scale, prompt_position=-1, max_new_tokens=max_new_tokens
        )
    if mode == "repeated":
        return runner.generate(
            list(PROMPTS),
            scale=scale,
            prompt_position=0,
            generation_period=1,
            max_new_tokens=max_new_tokens,
        )
    if mode == "periodic":
        return runner.generate(
            list(PROMPTS),
            scale=scale,
            prompt_position=0,
            generation_period=period,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--period", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "tiny"
        args.max_new_tokens = 3
    model, tokenizer = load_runtime(args.model)
    layers = gdn_layers(model)
    device = next(model.parameters()).device
    deltas = {
        layer: torch.randn(
            model.model.layers[layer].linear_attn.num_v_heads,
            model.model.layers[layer].linear_attn.head_k_dim,
            model.model.layers[layer].linear_attn.head_v_dim,
            device=device,
        )
        for layer in layers
    }
    runner = Runner(model, tokenizer, layers, deltas, normalize=False)
    rows = []
    for mode in MODES:
        tokens = generate(runner, mode, args.scale, args.period, args.max_new_tokens)
        if not torch.isfinite(tokens.float()).all():
            raise SystemExit(f"{mode} produced non-finite tokens")
        for index, row in enumerate(tokens):
            rows.append(
                {
                    "mode": mode,
                    "prompt": PROMPTS[index],
                    "response": tokenizer.decode(row, skip_special_tokens=True),
                    "scale": args.scale,
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
