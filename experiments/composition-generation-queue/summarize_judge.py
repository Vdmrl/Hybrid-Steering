"""Merge Judge formats and summarize composition conditions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

FEATURES = ("principled_candor", "concrete_language", "casualness", "optimism")
SHORT = {
    "candor": "principled_candor",
    "concrete": "concrete_language",
    "casual": "casualness",
    "optimism": "optimism",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-results", type=Path, required=True)
    parser.add_argument("--compact-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_ci(values: list[float]) -> dict[str, float | int]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        radius = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        radius = 1.96 * math.sqrt(variance / len(values))
    return {
        "n": len(values),
        "mean": round(mean, 4),
        "ci95_low": round(mean - radius, 4),
        "ci95_high": round(mean + radius, 4),
    }


def proportion_ci(successes: int, total: int) -> dict[str, float | int]:
    p = successes / total
    z = 1.96
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return {
        "successes": successes,
        "n": total,
        "rate": round(p, 4),
        "ci95_low": round(center - radius, 4),
        "ci95_high": round(center + radius, 4),
    }


def phase_from(path: Path) -> str:
    return path.stem.split("-", 1)[0]


def load_results(roots: tuple[Path, ...]) -> dict[tuple[str, str, str, str], dict]:
    # Compact is loaded first; audited JSON overrides it when both exist.
    merged = {}
    for root in roots:
        for path in sorted(root.glob("*.jsonl")):
            if path.name.endswith(".failures.jsonl"):
                continue
            phase = phase_from(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (phase, row["prompt_id"], row["answer_id"], row["feature"])
                merged[key] = row
    return merged


def mask_features(answer_id: str) -> tuple[str, ...]:
    bits = answer_id.rsplit("_", 1)[-1]
    mask = int(bits, 2)
    return tuple(feature for index, feature in enumerate(FEATURES) if mask & 1 << index)


def active_features(phase: str, answer_id: str) -> tuple[str, ...]:
    if phase in {"svd", "norm"}:
        return mask_features(answer_id)
    if phase == "activation":
        if answer_id == "baseline":
            return FEATURES
        suffix = answer_id.rsplit("_", 1)[-1]
        return FEATURES if suffix == "all4" else (SHORT[suffix],)
    if phase == "joy":
        return ("joy", "optimism") if answer_id.endswith("_optimism") else ("joy",)
    raise ValueError(f"unknown phase: {phase}")


def summarize(rows: dict[tuple[str, str, str, str], dict]) -> dict:
    by_condition: dict[tuple[str, str], dict[tuple[str, str], float]] = defaultdict(
        dict
    )
    for (phase, prompt_id, answer_id, feature), row in rows.items():
        by_condition[(phase, answer_id)][(prompt_id, feature)] = row["trait_score"]

    conditions = []
    for (phase, answer_id), scores in sorted(by_condition.items()):
        active = active_features(phase, answer_id)
        prompt_ids = sorted({prompt_id for prompt_id, _ in scores})
        complete = [
            prompt_id
            for prompt_id in prompt_ids
            if all((prompt_id, feature) in scores for feature in active)
        ]
        feature_metrics = {
            feature: mean_ci([scores[(prompt_id, feature)] for prompt_id in complete])
            for feature in active
        }
        minimums = [
            min(scores[(prompt_id, feature)] for feature in active)
            for prompt_id in complete
        ]
        conditions.append(
            {
                "phase": phase,
                "condition": answer_id,
                "active_features": list(active),
                "n_joint": len(complete),
                "feature_scores": feature_metrics,
                "minimum_active_score": mean_ci(minimums),
                "all_active_ge_4": proportion_ci(
                    sum(value >= 4 for value in minimums), len(minimums)
                ),
            }
        )
    return {"schema_version": "1.0", "conditions": conditions}


def markdown(summary: dict) -> str:
    lines = [
        "# Composition Judge summary",
        "",
        (
            "Audited JSON scores are preferred where available; compact one-digit "
            "scores fill only missing evaluations."
        ),
        "",
        "| Phase | Condition | Active features | N | All ≥4 | Mean minimum |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        joint = row["all_active_ge_4"]
        minimum = row["minimum_active_score"]
        lines.append(
            f"| {row['phase']} | `{row['condition']}` | "
            f"{', '.join(row['active_features'])} | {row['n_joint']} | "
            f"{joint['rate']:.1%} [{joint['ci95_low']:.1%}, "
            f"{joint['ci95_high']:.1%}] | {minimum['mean']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    assert mask_features("per_r1_0101") == (
        "principled_candor",
        "casualness",
    )
    assert active_features("joy", "joy_a2_optimism") == ("joy", "optimism")
    assert proportion_ci(5, 10)["rate"] == 0.5


def main() -> None:
    args = arguments()
    self_test()
    rows = load_results((args.compact_results, args.json_results))
    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(markdown(summary), encoding="utf-8")
    print(f"summarized {len(rows)} scores into {len(summary['conditions'])} conditions")


if __name__ == "__main__":
    main()
