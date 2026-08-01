from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def grouped_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    scenarios: dict[str, str] = {}
    for row in rows:
        source_id = str(row["source_id"])
        scenarios[source_id] = row["scenario"]
        grouped[source_id].append(
            {"answer_id": row["condition"], "text": row["response"]}
        )
    return [
        {
            "prompt_id": source_id,
            "scenario": scenarios[source_id],
            "answers": answers,
            "metadata": {"experiment": "urgent-russian-refusal-composition"},
        }
        for source_id, answers in grouped.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for value in grouped_rows(rows):
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
