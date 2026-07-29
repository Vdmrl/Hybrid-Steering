"""Summarize the candor/calm/concrete/optimism factorial extension."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

FEATURES = ("candor", "calm", "concrete", "optimism")


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score(row: dict, field: str = "trait_winner_answer_id") -> int:
    winner = row[field]
    return 1 if winner == "on" else -1 if winner == "off" else 0


def edges_for(
    feature: str, output_dir: Path, base_output_dir: Path, field: str
) -> dict[tuple[str, str], int]:
    result = {}
    if feature != "optimism":
        for row in rows(
            base_output_dir / "judge" / f"main-{feature}.aggregated.jsonl"
        ):
            mask = row["prompt_id"].rsplit(":", 1)[-1]
            if int(mask, 2) < 8:
                source = ":".join(row["prompt_id"].split(":")[1:-1])
                result[(source, mask)] = score(row, field)
    for row in rows(output_dir / "judge" / f"main-{feature}.aggregated.jsonl"):
        mask = row["prompt_id"].rsplit(":", 1)[-1]
        source = ":".join(row["prompt_id"].split(":")[1:-1])
        result[(source, mask)] = score(row, field)
    return result


def bootstrap(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [draws[250], draws[9750]]


def main_effect(edges: dict[tuple[str, str], int], seed: int) -> dict:
    by_prompt: dict[str, list[int]] = defaultdict(list)
    for (source, _), value in edges.items():
        by_prompt[source].append(value)
    values = [mean(items) for items in by_prompt.values()]
    return {
        "prompts": len(values),
        "comparisons": sum(len(items) for items in by_prompt.values()),
        "effect": mean(values),
        "ci95": bootstrap(values, seed),
    }


def interaction(
    edges: dict[tuple[str, str], int],
    target_index: int,
    context_index: int,
    seed: int,
) -> dict:
    sources = sorted({source for source, _ in edges})
    values = []
    for source in sources:
        off, on = [], []
        for mask in range(16):
            if mask & (1 << target_index):
                continue
            value = edges.get((source, f"{mask:04b}"))
            if value is None:
                continue
            (on if mask & (1 << context_index) else off).append(value)
        if off and on:
            values.append(mean(on) - mean(off))
    return {
        "prompts": len(values),
        "difference_in_differences": mean(values),
        "ci95": bootstrap(values, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-output-dir", type=Path, required=True)
    args = parser.parse_args()

    trait_edges = {
        feature: edges_for(
            feature, args.output_dir, args.base_output_dir, "trait_winner_answer_id"
        )
        for feature in FEATURES
    }
    quality_edges = {
        feature: edges_for(
            feature, args.output_dir, args.base_output_dir, "quality_winner_answer_id"
        )
        for feature in FEATURES
    }
    summary = {
        "main_effects": {
            feature: main_effect(trait_edges[feature], 20260729 + index)
            for index, feature in enumerate(FEATURES)
        },
        "quality_effects": {
            feature: main_effect(quality_edges[feature], 20260829 + index)
            for index, feature in enumerate(FEATURES)
        },
        "interactions": {},
    }
    for target_index, target in enumerate(FEATURES):
        for context_index, context in enumerate(FEATURES):
            if target == context:
                continue
            summary["interactions"][f"{target}|context={context}"] = interaction(
                trait_edges[target],
                target_index,
                context_index,
                20260929 + target_index * 4 + context_index,
            )
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
