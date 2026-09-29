"""Context-length steering modes on one saved direction.

Score each response with ``concept_detector``. ``--feature`` defaults to the
direction target.

``initial`` writes the direction before the first real token.
``prompt-end`` writes it at the last prompt token.
``repeated`` and ``periodic`` refresh it during generation.
The initial state is zero, so these modes use ``normalize=False``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hybrid_steering import Runner, concept_detector, load_direction, load_runtime
from hybrid_steering.judge.config import repo_root
from hybrid_steering.runtime import import_path, write_jsonl

MODES = ("initial", "prompt-end", "repeated", "periodic")
SCALES = (1.25, 1.5)


def simple_questions(count: int, seed: int) -> list[dict[str, str]]:
    module = import_path(repo_root() / "experiments/forgetting/questions.py")
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
    parser.add_argument("--feature", help="defaults to the direction target")
    parser.add_argument("--scales", type=float, nargs="+", default=SCALES)
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--period", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prompts = [row["question"] for row in simple_questions(args.questions, args.seed)]
    direction, manifest, _, _ = load_direction(args.direction)
    model, tokenizer = load_runtime(args.model)
    runner = Runner.from_direction(
        model, tokenizer, direction, rank=args.rank or None, normalize=False
    )
    detector = concept_detector(args.feature or manifest.target)
    rows = []
    for scale in args.scales:
        for mode in MODES:
            tokens = generate(runner, prompts, mode, scale, args.period, args.max_new_tokens)
            if not torch.isfinite(tokens.float()).all():
                raise SystemExit(f"{mode} produced non-finite tokens")
            for prompt, row in zip(prompts, tokens, strict=True):
                text = tokenizer.decode(row, skip_special_tokens=True)
                rows.append(
                    {
                        "mode": mode,
                        "prompt": prompt,
                        "response": text,
                        "scale": scale,
                        "label": detector.label(text),
                        "concept_score": int(detector.detects(text)),
                        "target": manifest.target,
                        "source": manifest.source,
                    }
                )
    write_jsonl(args.output / "responses.jsonl", rows)
    print(f"wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
