"""How filler tokens after an initial-state intervention change a concept score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_steering import (
    DirectionManifest,
    Runner,
    collect_direction,
    concept_detector,
    gdn_layers,
    load_direction,
    load_runtime,
    save_direction,
    token_prefixes,
    truncate_direction,
)
from hybrid_steering.runtime import chat_prompts

QUESTIONS = (
    "What might happen if someone misses the last bus home?",
    "How would you decide whether to take an umbrella?",
)
FILLER = "The warehouse logged each crate, label, and route before the truck left. " * 8


def direction_for(args: argparse.Namespace, output: Path) -> dict[int, torch.Tensor]:
    if args.direction:
        direction, manifest, _, _ = load_direction(args.direction)
        print(f"loaded {manifest.target} - {manifest.source}", flush=True)
        return direction
    model, tokenizer = load_runtime(args.model)
    layers = gdn_layers(model)
    runner = Runner(model, tokenizer, layers)
    pairs = (
        ("Он говорит фактами и называет даты.", "Он придумывает сцену, героя и диалог."),
        ("Отчёт перечисляет наблюдения.", "Рассказ описывает внутренние мысли героя."),
    )
    collected = collect_direction(runner, pairs[:1] if args.smoke else pairs)
    manifest = DirectionManifest(
        model_id=args.model,
        target="factual reporting",
        source="fictional narrative",
        example_ids=["pair-0"],
        decoder_layer_indices=sorted(collected.delta),
        state_shapes={layer: list(tensor.shape) for layer, tensor in collected.delta.items()},
    )
    save_direction(output / "direction", collected.delta, manifest)
    return collected.delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--feature", default="fictional_narrative")
    parser.add_argument("--questions", type=int, default=2)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[0, 32])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "tiny"
        args.questions = 1
        args.scales = [0.5]
        args.prefix_lengths = [0, 4]
        args.max_new_tokens = 2
        args.rank = 1
    detector = concept_detector(args.feature, verdict="1" if args.smoke else None)
    model, tokenizer = load_runtime(args.model)
    deltas = truncate_direction(direction_for(args, args.output), args.rank or None)
    device = next(model.parameters()).device
    runner = Runner(
        model,
        tokenizer,
        sorted(deltas),
        {layer: tensor.to(device) for layer, tensor in deltas.items()},
        normalize=True,
    )
    questions = list(QUESTIONS[: args.questions])
    prefixes = token_prefixes(tokenizer, FILLER, args.prefix_lengths)
    rows = []
    for length, prefix in prefixes.items():
        texts = chat_prompts(tokenizer, questions, prefix)
        baseline = runner.generate(texts, prompt_position=None, max_new_tokens=args.max_new_tokens)
        for scale in args.scales:
            steered = runner.generate(
                texts, scale=scale, prompt_position=0, max_new_tokens=args.max_new_tokens
            )
            for condition, tokens in (("baseline", baseline), ("steered", steered)):
                for question, row_tokens in zip(questions, tokens, strict=True):
                    response = tokenizer.decode(row_tokens, skip_special_tokens=True)
                    rows.append(
                        {
                            "prefix_length": length,
                            "scale": scale,
                            "condition": condition,
                            "question": question,
                            "response": response,
                            "concept_score": int(detector.detects(response, question=question)),
                            "target": detector.target,
                            "source": detector.source,
                        }
                    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print(f"wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
