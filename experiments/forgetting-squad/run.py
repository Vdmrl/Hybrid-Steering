"""SQuAD-style retention: language detection plus factual equivalence.

Smoke uses one local question instead of the SQuAD download. ``--squad`` loads
the validation split when a full run is intended.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_steering import (
    AnswerEquivalence,
    DirectionManifest,
    Runner,
    collect_direction,
    concept_detector,
    gdn_layers,
    load_runtime,
    save_direction,
    token_prefixes,
    truncate_direction,
)
from hybrid_steering.judge.config import load_configs, repo_root
from hybrid_steering.judge.runner import complete_text
from hybrid_steering.runtime import chat_prompts

QUESTIONS = (
    {"id": "local-1", "question": "What is the capital of France?"},
    {"id": "local-2", "question": "Who wrote Hamlet?"},
)
FILLER = "Shipping documents listed the part number, batch, and quantity. " * 8
PAIRS = (
    ("Это ответ на русском языке про город.", "This is an English answer about a city."),
    ("Завтра в Москве будет снег.", "Tomorrow it will snow in Moscow."),
)


def questions(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.squad:
        return [
            {"id": row["id"], "question": row["question"]} for row in QUESTIONS[: args.questions]
        ]
    from datasets import load_dataset

    result = []
    for row in load_dataset("rajpurkar/squad", split="validation").shuffle(seed=args.seed):
        if row["question"]:
            result.append({"id": str(row["id"]), "question": str(row["question"])})
        if len(result) == args.questions:
            return result
    raise RuntimeError(f"found only {len(result)} SQuAD questions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--questions", type=int, default=2)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[0, 32])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--squad", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "tiny"
        args.questions = 1
        args.scales = [0.5]
        args.prefix_lengths = [0, 4]
        args.max_new_tokens = 2
        args.squad = False
    model, tokenizer = load_runtime(args.model)
    layers = gdn_layers(model)
    collector = Runner(model, tokenizer, layers)
    collected = collect_direction(collector, PAIRS[:1] if args.smoke else PAIRS)
    manifest = DirectionManifest(
        model_id=args.model,
        target=args.language,
        source="en",
        example_ids=["pair-0"],
        decoder_layer_indices=sorted(collected.delta),
        state_shapes={layer: list(tensor.shape) for layer, tensor in collected.delta.items()},
    )
    save_direction(args.output / "direction", collected.delta, manifest)
    device = next(model.parameters()).device
    deltas = {
        layer: tensor.to(device)
        for layer, tensor in truncate_direction(collected.delta, args.rank or None).items()
    }
    runner = Runner(model, tokenizer, sorted(deltas), deltas, normalize=True)
    detector = concept_detector(args.language)
    equivalence = AnswerEquivalence()
    _, judge_config = load_configs(repo_root())

    def judge_text(text: str) -> str:
        if args.smoke:
            return "<verdict>1</verdict>"
        return complete_text(
            text,
            model=judge_config.model,
            base_url=judge_config.base_url,
            extra=judge_config.generation.request_extras,
        )

    rows = []
    examples = questions(args)
    prefixes = token_prefixes(tokenizer, FILLER, args.prefix_lengths)
    asked = [row["question"] for row in examples]
    for length, prefix in prefixes.items():
        texts = chat_prompts(tokenizer, asked, prefix)
        baseline_tokens = runner.generate(
            texts, prompt_position=None, max_new_tokens=args.max_new_tokens
        )
        baseline_text = [tokenizer.decode(row, skip_special_tokens=True) for row in baseline_tokens]
        for scale in args.scales:
            steered_tokens = runner.generate(
                texts, scale=scale, prompt_position=0, max_new_tokens=args.max_new_tokens
            )
            for example, base, steered in zip(examples, baseline_text, steered_tokens, strict=True):
                response = tokenizer.decode(steered, skip_special_tokens=True)
                verdict, raw = equivalence.score(
                    example["question"],
                    response,
                    base,
                    judge_text,
                )
                rows.append(
                    {
                        "id": example["id"],
                        "prefix_length": length,
                        "scale": scale,
                        "question": example["question"],
                        "baseline": base,
                        "response": response,
                        "language": detector.label(response),
                        "target_language": detector.detects(response),
                        "equivalent": verdict,
                        "equivalence_raw": raw,
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print(f"wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
