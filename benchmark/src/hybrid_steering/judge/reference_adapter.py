"""Export saved answers for Hybrid-Steering's 1–5 Judge without leaking conditions."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from .io import read_jsonl, write


def identifier(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def convert(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Return blind reference input and a private source binding."""
    scenarios: dict[str, dict] = {}
    bindings = []
    seen_answers: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        prompt, response = row.get("prompt"), row.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(f"Row {index} needs string prompt and response")
        prompt_id = identifier(prompt)
        answer_id = identifier(response)
        item = scenarios.setdefault(prompt_id, {
            "prompt_id": prompt_id, "scenario": prompt, "answers": [],
        })
        if item["scenario"] != prompt:
            raise ValueError("Prompt ID collision")
        pair = (prompt_id, answer_id)
        if pair not in seen_answers:
            item["answers"].append({"answer_id": answer_id, "text": response})
            seen_answers.add(pair)
        bindings.append({
            "prompt_id": prompt_id,
            "answer_id": answer_id,
            "source_index": index,
            "source_id": row.get("key", row.get("id", index)),
            "condition": {key: row.get(key) for key in ("method", "layer", "strength")},
        })
    blind = list(scenarios.values())
    rng = random.Random(seed)
    rng.shuffle(blind)
    for item in blind:
        rng.shuffle(item["answers"])
    return blind, bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.resolve() == args.bindings.resolve():
        parser.error("Blind input and private bindings must be separate files")
    blind, bindings = convert(read_jsonl(args.input), args.seed)
    write(args.output, blind)
    write(args.bindings, bindings)
    print(json.dumps({"scenarios": len(blind), "bindings": len(bindings)}))


if __name__ == "__main__":
    main()
