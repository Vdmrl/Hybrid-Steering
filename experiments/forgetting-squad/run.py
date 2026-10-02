"""SQuAD retention after a filler prefix: language detection and equivalence.

Questions come from the SQuAD validation split. The filler is one passage from
``forgetting/fillers.py``. The direction is a saved target-minus-source artifact.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset
from equivalence import prompt as equivalence_prompt
from equivalence import verdict

from hybrid_steering import Runner, concept_detector, load_direction, load_runtime, token_prefixes
from hybrid_steering.judge import complete_batch
from hybrid_steering.judge.config import repo_root
from hybrid_steering.runtime import collect_steered_rows, import_path, write_jsonl

PREFIX_LENGTHS = (0, 32, 64, 128, 256, 512, 1024, 2048, 4096)
SCALES = tuple(index / 2 for index in range(2, 11))


def filler_passages() -> list[str]:
    module = import_path(repo_root() / "experiments/forgetting/fillers.py")
    return module.FILLERS


def squad_questions(questions: int, seed: int) -> list[dict[str, str]]:
    result = []
    for row in load_dataset("rajpurkar/squad", split="validation").shuffle(seed=seed):
        if row["question"]:
            result.append({"source_id": str(row["id"]), "question": str(row["question"])})
        if len(result) == questions:
            return result
    raise RuntimeError(f"found only {len(result)} SQuAD questions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--scales", type=float, nargs="+", default=SCALES)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=PREFIX_LENGTHS)
    parser.add_argument("--filler-index", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefill-token-budget", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--rank", type=int, default=2)
    args = parser.parse_args()
    if args.questions < 1:
        parser.error("--questions must be positive")
    passages = filler_passages()
    filler_index = (
        args.filler_index
        if args.filler_index is not None
        else random.Random(args.seed).randrange(len(passages))
    )
    direction, manifest, _, _ = load_direction(args.direction)
    model, tokenizer = load_runtime(args.model)
    runner = Runner.from_direction(
        model, tokenizer, direction, rank=args.rank or None, normalize=True
    )
    detector = concept_detector(args.language)
    examples = squad_questions(args.questions, args.seed)
    prefixes = token_prefixes(tokenizer, passages[filler_index], args.prefix_lengths)

    def build_row(length, example, base, scale, response):
        return {
            "source_id": example["source_id"],
            "prefix_length": length,
            "filler_index": filler_index,
            "scale": scale,
            "question": example["question"],
            "baseline": base,
            "response": response,
            "language": detector.label(response),
            "target_language": int(detector.detects(response)),
            "direction": f"{manifest.target} - {manifest.source}",
        }

    rows = collect_steered_rows(
        runner,
        tokenizer,
        examples,
        prefixes,
        args.scales,
        batch_size=args.batch_size,
        token_budget=args.prefill_token_budget,
        max_new_tokens=args.max_new_tokens,
        build_row=build_row,
    )
    raws = complete_batch(
        [
            [
                {
                    "role": "user",
                    "content": equivalence_prompt(
                        row["question"], row["response"], row["baseline"]
                    ),
                }
            ]
            for row in rows
        ],
        max_tokens=32,
    )
    for row, raw in zip(rows, raws, strict=True):
        row["equivalent"] = verdict(raw)
        row["equivalence_raw"] = raw
    write_jsonl(args.output / "rows.jsonl", rows)
    print(f"wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
