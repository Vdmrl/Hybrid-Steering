"""Context-length steering modes on one direction.

``initial`` writes the direction before the first real token.
``prompt-end`` writes it at the last prompt token.
``repeated`` and ``periodic`` refresh it during generation.
The initial state is zero, so these modes use ``normalize=False``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from hybrid_steering import Runner, load_direction, load_runtime, truncate_direction

MODES = ("initial", "prompt-end", "repeated", "periodic")
SCALES = (1.25, 1.5)


def simple_questions(count: int, seed: int) -> list[dict[str, str]]:
    path = Path(__file__).resolve().parents[1] / "forgetting" / "questions.py"
    spec = importlib.util.spec_from_file_location("forgetting_questions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simple_questions(count, seed)


def generate(
    runner: Runner,
    prompts: list[str],
    mode: str,
    scale: float,
    period: int,
    max_new_tokens: int,
) -> torch.Tensor:
    if mode == "initial":
        return runner.generate(
            prompts, scale=scale, prompt_position=0, max_new_tokens=max_new_tokens
        )
    if mode == "prompt-end":
        return runner.generate(
            prompts, scale=scale, prompt_position=-1, max_new_tokens=max_new_tokens
        )
    if mode == "repeated":
        return runner.generate(
            prompts,
            scale=scale,
            prompt_position=0,
            generation_period=1,
            max_new_tokens=max_new_tokens,
        )
    if mode == "periodic":
        return runner.generate(
            prompts,
            scale=scale,
            prompt_position=0,
            generation_period=period,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--scales", type=float, nargs="+", default=SCALES)
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--period", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prompts = [row["question"] for row in simple_questions(args.questions, args.seed)]
    direction, _, _, _ = load_direction(args.direction)
    model, tokenizer = load_runtime(args.model)
    device = next(model.parameters()).device
    deltas = {
        layer: tensor.to(device)
        for layer, tensor in truncate_direction(direction, args.rank or None).items()
    }
    runner = Runner(model, tokenizer, sorted(deltas), deltas, normalize=False)
    rows = []
    for scale in args.scales:
        for mode in MODES:
            tokens = generate(runner, prompts, mode, scale, args.period, args.max_new_tokens)
            if not torch.isfinite(tokens.float()).all():
                raise SystemExit(f"{mode} produced non-finite tokens")
            for prompt, row in zip(prompts, tokens, strict=True):
                rows.append(
                    {
                        "mode": mode,
                        "prompt": prompt,
                        "response": tokenizer.decode(row, skip_special_tokens=True),
                        "scale": scale,
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
