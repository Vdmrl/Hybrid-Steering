"""How filler-token distance weakens one concept intervention.

The filler is one passage from ``fillers.FILLERS``, cut to the prefix lengths.
Questions are a seeded sample of the simple-question bank. The direction is a
saved target-minus-source artifact.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fillers import FILLERS
from questions import SIMPLE_QUESTIONS, simple_questions

from hybrid_steering import (
    Runner,
    concept_detector,
    load_direction,
    load_runtime,
    token_prefixes,
    truncate_direction,
)
from hybrid_steering.runtime import chat_prompts

PREFIX_LENGTHS = (0, 32, 64, 128, 256, 512, 1024, 2048, 4096)
SCALES = tuple(index / 2 for index in range(2, 11))


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--scales", type=float, nargs="+", default=SCALES)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=PREFIX_LENGTHS)
    parser.add_argument("--filler-index", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefill-token-budget", type=int, default=65536)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--rank", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.questions <= len(SIMPLE_QUESTIONS):
        parser.error(f"--questions must be in [1, {len(SIMPLE_QUESTIONS)}]")
    filler_index = (
        args.filler_index
        if args.filler_index is not None
        else random.Random(args.seed).randrange(len(FILLERS))
    )
    if not 0 <= filler_index < len(FILLERS):
        parser.error(f"--filler-index must be in [0, {len(FILLERS) - 1}]")
    direction, manifest, _, _ = load_direction(args.direction)
    model, tokenizer = load_runtime(args.model)
    deltas = {
        layer: tensor.to(next(model.parameters()).device)
        for layer, tensor in truncate_direction(direction, args.rank or None).items()
    }
    runner = Runner(model, tokenizer, sorted(deltas), deltas, normalize=True)
    detector = concept_detector(args.feature)
    examples = simple_questions(args.questions, args.seed)
    prefixes = token_prefixes(tokenizer, FILLERS[filler_index], args.prefix_lengths)
    rows = []
    for length, prefix in prefixes.items():
        width = min(args.batch_size, max(1, args.prefill_token_budget // max(length, 1)))
        for batch in batched(examples, width):
            questions = [row["question"] for row in batch]
            texts = chat_prompts(tokenizer, questions, prefix)
            baseline = runner.generate(
                texts, prompt_position=None, max_new_tokens=args.max_new_tokens
            )
            baseline_text = [tokenizer.decode(row, skip_special_tokens=True) for row in baseline]
            for scale in args.scales:
                steered = runner.generate(
                    texts, scale=scale, prompt_position=0, max_new_tokens=args.max_new_tokens
                )
                for example, base, row_tokens in zip(batch, baseline_text, steered, strict=True):
                    response = tokenizer.decode(row_tokens, skip_special_tokens=True)
                    rows.append(
                        {
                            "source_id": example["source_id"],
                            "question": example["question"],
                            "prefix_length": length,
                            "filler_index": filler_index,
                            "scale": scale,
                            "baseline": base,
                            "response": response,
                            "concept_score": int(
                                detector.detects(response, question=example["question"])
                            ),
                            "target": manifest.target,
                            "source": manifest.source,
                        }
                    )
            print(f"prefix {length}: {len(rows)} rows", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print(f"wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
