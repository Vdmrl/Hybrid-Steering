"""SQuAD retention after a filler prefix: language detection and equivalence.

Questions come from the SQuAD validation split. The filler is one passage from
``forgetting/fillers.py``. The direction is a saved target-minus-source artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

from datasets import load_dataset

from hybrid_steering import (
    AnswerEquivalence,
    Runner,
    concept_detector,
    load_direction,
    load_runtime,
    token_prefixes,
    truncate_direction,
)
from hybrid_steering.judge.config import load_configs, repo_root
from hybrid_steering.judge.runner import complete_text
from hybrid_steering.runtime import chat_prompts

PREFIX_LENGTHS = (0, 32, 64, 128, 256, 512, 1024, 2048, 4096)
SCALES = tuple(index / 2 for index in range(2, 11))


def filler_passages() -> list[str]:
    path = Path(__file__).resolve().parents[1] / "forgetting" / "fillers.py"
    spec = importlib.util.spec_from_file_location("forgetting_fillers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FILLERS


def squad_questions(questions: int, seed: int) -> list[dict[str, str]]:
    result = []
    for row in load_dataset("rajpurkar/squad", split="validation").shuffle(seed=seed):
        if row["question"]:
            result.append({"source_id": str(row["id"]), "question": str(row["question"])})
        if len(result) == questions:
            return result
    raise RuntimeError(f"found only {len(result)} SQuAD questions")


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


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
    deltas = {
        layer: tensor.to(next(model.parameters()).device)
        for layer, tensor in truncate_direction(direction, args.rank or None).items()
    }
    runner = Runner(model, tokenizer, sorted(deltas), deltas, normalize=True)
    detector = concept_detector(args.language)
    equivalence = AnswerEquivalence()
    _, judge_config = load_configs(repo_root())
    examples = squad_questions(args.questions, args.seed)
    prefixes = token_prefixes(tokenizer, passages[filler_index], args.prefix_lengths)
    rows = []
    for length, prefix in prefixes.items():
        width = min(args.batch_size, max(1, args.prefill_token_budget // max(length, 1)))
        for batch in batched(examples, width):
            questions = [row["question"] for row in batch]
            texts = chat_prompts(tokenizer, questions, prefix)
            baseline_tokens = runner.generate(
                texts, prompt_position=None, max_new_tokens=args.max_new_tokens
            )
            baseline_text = [
                tokenizer.decode(row, skip_special_tokens=True) for row in baseline_tokens
            ]
            for scale in args.scales:
                steered_tokens = runner.generate(
                    texts, scale=scale, prompt_position=0, max_new_tokens=args.max_new_tokens
                )
                for example, base, steered in zip(
                    batch, baseline_text, steered_tokens, strict=True
                ):
                    response = tokenizer.decode(steered, skip_special_tokens=True)
                    verdict, raw = equivalence.score(
                        example["question"],
                        response,
                        base,
                        lambda text: complete_text(
                            text,
                            model=judge_config.model,
                            base_url=judge_config.base_url,
                            extra=judge_config.generation.request_extras,
                        ),
                    )
                    rows.append(
                        {
                            "source_id": example["source_id"],
                            "prefix_length": length,
                            "filler_index": filler_index,
                            "scale": scale,
                            "question": example["question"],
                            "baseline": base,
                            "response": response,
                            "language": detector.label(response),
                            "target_language": detector.detects(response),
                            "equivalent": verdict,
                            "equivalence_raw": raw,
                            "direction": f"{manifest.target} - {manifest.source}",
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
