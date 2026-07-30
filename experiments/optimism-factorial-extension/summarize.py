"""Summarize the candor/concrete/casual/optimism factorial extension."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

FEATURES = ("candor", "concrete", "casual", "optimism")


def old_to_new_mask(old_mask: int) -> int:
    return (old_mask & 1) | ((old_mask & 4) >> 1) | ((old_mask & 8) >> 1)


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
        for row in rows(base_output_dir / "judge" / f"main-{feature}.aggregated.jsonl"):
            old_mask = int(row["prompt_id"].rsplit(":", 1)[-1], 2)
            if not old_mask & 0b0010:
                source = ":".join(row["prompt_id"].split(":")[1:-1])
                result[(source, f"{old_to_new_mask(old_mask):04b}")] = score(row, field)
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


def composition_effects(
    edges: dict[tuple[str, str], int],
    target_index: int,
    seed: int,
) -> tuple[dict, dict]:
    sources = sorted({source for source, _ in edges})
    contexts, by_size = {}, defaultdict(lambda: defaultdict(list))
    for mask in range(16):
        if mask & (1 << target_index):
            continue
        values = [
            edges[(source, f"{mask:04b}")]
            for source in sources
            if (source, f"{mask:04b}") in edges
        ]
        active = [
            feature
            for index, feature in enumerate(FEATURES)
            if index != target_index and mask & (1 << index)
        ]
        contexts["+".join(active) or "none"] = {
            "effect": mean(values),
            "ci95": bootstrap(values, seed + mask),
        }
        for source in sources:
            key = (source, f"{mask:04b}")
            if key in edges:
                by_size[len(active)][source].append(edges[key])
    sizes = {}
    for size, prompts in by_size.items():
        values = [mean(items) for items in prompts.values()]
        sizes[str(size)] = {
            "effect": mean(values),
            "ci95": bootstrap(values, seed + 100 + size),
        }
    return contexts, sizes


def joint_compositions(
    edges: dict[str, dict[tuple[str, str], int]],
) -> dict:
    feature_index = {feature: index for index, feature in enumerate(FEATURES)}
    source_ids = sorted(
        set.intersection(
            *({source_id for source_id, _ in edges[feature]} for feature in FEATURES)
        )
    )
    result = {}
    for size in range(1, len(FEATURES) + 1):
        for active in combinations(FEATURES, size):
            active_mask = sum(1 << feature_index[feature] for feature in active)
            confirmed, reversed_counts = [], []
            for source_id in source_ids:
                values = [
                    edges[feature].get(
                        (
                            source_id,
                            f"{active_mask ^ (1 << feature_index[feature]):04b}",
                        )
                    )
                    for feature in active
                ]
                if any(value is None for value in values):
                    continue
                confirmed.append(sum(value > 0 for value in values))
                reversed_counts.append(sum(value < 0 for value in values))
            if not confirmed:
                continue
            full = [value == size for value in confirmed]
            no_reversal = [value == 0 for value in reversed_counts]
            result["+".join(active)] = {
                "features": list(active),
                "prompts": len(confirmed),
                "confirmed_distribution": {
                    str(value): confirmed.count(value) for value in range(size + 1)
                },
                "all_confirmed": sum(full),
                "all_confirmed_rate": mean(full),
                "all_confirmed_ci95": bootstrap(full, 20261229 + active_mask),
                "mean_confirmed": mean(confirmed),
                "no_reversal": sum(no_reversal),
                "no_reversal_rate": mean(no_reversal),
            }
    return result


def language_effect(path: Path, seed: int) -> dict:
    complete = [row for row in rows(path) if row["status"] == "complete"]
    trait = [score(row) for row in complete]
    quality = [score(row, "quality_winner_answer_id") for row in complete]
    return {
        "n": len(complete),
        "trait_effect": mean(trait),
        "trait_ci95": bootstrap(trait, seed),
        "quality_effect": mean(quality),
        "quality_ci95": bootstrap(quality, seed + 100),
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
        "effects_by_context": {},
        "effects_by_context_size": {},
        "joint_compositions": joint_compositions(trait_edges),
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
        (
            summary["effects_by_context"][target],
            summary["effects_by_context_size"][target],
        ) = composition_effects(
            trait_edges[target],
            target_index,
            20261029 + target_index * 100,
        )
    language_dir = args.output_dir / "language"
    summary["optimism_french"] = {
        name: language_effect(
            language_dir / "judge" / f"{name}.raw.aggregated.jsonl",
            20261129 + index,
        )
        for index, name in enumerate(
            (
                "optimism_single",
                "optimism_with_french",
                "french_single",
                "french_with_optimism",
            )
        )
    }
    summary["optimism_french"]["deterministic_language"] = json.loads(
        (language_dir / "language_summary.json").read_text(encoding="utf-8")
    )
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
