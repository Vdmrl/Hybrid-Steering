"""Summarize the candor/French 2x2 composition experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

COMPARISONS = (
    "candor_single",
    "candor_with_french",
    "french_single",
    "french_with_candor",
)


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(path: Path, seed: int) -> dict:
    complete = [row for row in rows(path) if row["status"] == "complete"]

    def score(row: dict, field: str) -> int:
        winner = row[field]
        return 1 if winner == "target" else -1 if winner == "control" else 0

    trait = [score(row, "trait_winner_answer_id") for row in complete]
    quality = [score(row, "quality_winner_answer_id") for row in complete]
    rng = random.Random(seed)

    def interval(values: list[int]) -> list[float]:
        draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10000))
        return [draws[250], draws[9750]]

    return {
        "n": len(complete),
        "trait_effect": mean(trait),
        "trait_ci95": interval(trait),
        "quality_effect": mean(quality),
        "quality_ci95": interval(quality),
        "trait_order_consistency": sum(
            row["trait_order_consistent"] is True for row in complete
        )
        / len(complete),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = {
        name: summarize(
            args.output_dir / "judge" / f"{name}.raw.aggregated.jsonl",
            20260729 + index,
        )
        for index, name in enumerate(COMPARISONS)
    }
    summary["deterministic_language"] = json.loads(
        (args.output_dir / "language_summary.json").read_text(encoding="utf-8")
    )
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
