"""Cluster-bootstrap pairwise effects and factorial interactions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

FEATURES = ("candor", "calm", "concrete", "casual")


def jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def score(row: dict) -> float:
    winner = row["trait_winner_answer_id"]
    return 1.0 if winner == "on" else -1.0 if winner == "off" else 0.0


def bootstrap(values: list[float], rng: random.Random, samples: int) -> list[float]:
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [draws[int(0.025 * samples)], draws[int(0.975 * samples)]]


def summarize_file(
    path: Path, rng: random.Random, samples: int
) -> tuple[dict, dict[tuple[str, str], float]]:
    by_prompt: dict[str, list[float]] = {}
    edges: dict[tuple[str, str], float] = {}
    consistency = []
    for row in jsonl(path):
        parts = row["prompt_id"].split(":")
        source_id = ":".join(parts[1:-1])
        comparison = parts[-1]
        value = score(row)
        by_prompt.setdefault(source_id, []).append(value)
        edges[source_id, comparison] = value
        consistency.append(row["trait_order_consistent"] is not False)
    prompt_effects = [mean(values) for values in by_prompt.values()]
    return (
        {
            "prompts": len(prompt_effects),
            "comparisons": sum(len(values) for values in by_prompt.values()),
            "signed_effect": mean(prompt_effects),
            "ci95": bootstrap(prompt_effects, rng, samples),
            "order_consistency": sum(consistency) / len(consistency),
        },
        edges,
    )


def interaction_effects(
    edges: dict[tuple[str, str], float],
    target_index: int,
    context_index: int,
) -> list[float]:
    source_ids = sorted({source_id for source_id, _ in edges})
    effects = []
    for source_id in source_ids:
        context_off, context_on = [], []
        for mask in range(16):
            if mask & (1 << target_index):
                continue
            value = edges.get((source_id, f"{mask:04b}"))
            if value is None:
                continue
            destination = context_on if mask & (1 << context_index) else context_off
            destination.append(value)
        if context_off and context_on:
            effects.append(mean(context_on) - mean(context_off))
    return effects


def main() -> None:
    args = arguments()
    rng = random.Random(args.seed)
    pairwise = {}
    main_edges = {}
    for path in sorted((args.output_dir / "judge").glob("*.aggregated.jsonl")):
        result, edges = summarize_file(path, rng, args.samples)
        pairwise[path.stem.removesuffix(".aggregated")] = result
        if path.stem.startswith("main-"):
            main_edges[path.stem.removeprefix("main-").removesuffix(".aggregated")] = (
                edges
            )

    interactions = {}
    for target_index, target in enumerate(FEATURES):
        if target not in main_edges:
            continue
        for context_index, context in enumerate(FEATURES):
            if target == context:
                continue
            values = interaction_effects(
                main_edges[target], target_index, context_index
            )
            interactions[f"{target}|context={context}"] = {
                "prompts": len(values),
                "difference_in_differences": mean(values),
                "ci95": bootstrap(values, rng, args.samples),
            }

    output = {
        "pairwise_effects": pairwise,
        "factorial_interactions": interactions,
        "interpretation": (
            "Positive signed effects favor answer_id=on. Interaction values show "
            "how another active feature changes the target feature's matched effect."
        ),
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
