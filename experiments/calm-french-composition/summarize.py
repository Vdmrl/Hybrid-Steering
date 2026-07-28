"""Summarize Judge v2 pairwise aggregates for the 2x2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    return {
        "n": len(rows),
        "complete": len(complete),
        "trait_target_wins": sum(
            row["trait_winner_answer_id"] == "target" for row in complete
        ),
        "trait_control_wins": sum(
            row["trait_winner_answer_id"] == "control" for row in complete
        ),
        "trait_ties": sum(row["trait_winner_answer_id"] == "tie" for row in complete),
        "trait_order_inconsistent": sum(
            row["trait_order_consistent"] is False for row in complete
        ),
        "quality_target_wins": sum(
            row["quality_winner_answer_id"] == "target" for row in complete
        ),
        "quality_control_wins": sum(
            row["quality_winner_answer_id"] == "control" for row in complete
        ),
        "quality_ties": sum(
            row["quality_winner_answer_id"] == "tie" for row in complete
        ),
    }


def main() -> None:
    args = arguments()
    judge_dir = args.output_dir / "judge"
    comparisons = (
        "calm_single",
        "calm_with_french",
        "french_single",
        "french_with_calm",
    )
    summary = {
        name: summarize(jsonl(judge_dir / f"{name}.raw.aggregated.jsonl"))
        for name in comparisons
    }
    summary["deterministic_language"] = json.loads(
        (args.output_dir / "language_summary.json").read_text(encoding="utf-8")
    )
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
