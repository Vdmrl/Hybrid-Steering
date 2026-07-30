"""Paired composition analysis for the second dashboard."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import summarize_judge as summary

SEED = 20260730
FEATURES = summary.FEATURES
METHODS = {
    "baseline": ("activation_holdout", "baseline"),
    "gdn_rank1": ("svd", "per_r1_1111"),
    "gdn_rank4": ("svd", "per_r4_1111"),
    "gdn_norm": ("norm", "norm_1111"),
    "activation_l10_a4": ("activation_holdout", "l10_a4_all4"),
    "activation_l20_a1": ("activation_holdout", "l20_a1_all4"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-results", type=Path, required=True)
    parser.add_argument("--compact-results", type=Path, required=True)
    parser.add_argument("--activation-holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def paired_ci(values: list[float], samples: int = 10_000) -> dict[str, float | int]:
    rng = random.Random(SEED)
    boot = [
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)
    ]
    return {
        "n": len(values),
        "difference": round(sum(values) / len(values), 4),
        "ci95_low": round(percentile(boot, 0.025), 4),
        "ci95_high": round(percentile(boot, 0.975), 4),
    }


def score_map(
    rows: dict[tuple[str, str, str, str], dict],
    phase: str,
    answer_id: str,
    prompt_ids: set[str],
) -> dict[str, list[float]]:
    result = {}
    for prompt_id in prompt_ids:
        scores = [
            rows.get((phase, prompt_id, answer_id, feature), {}).get("trait_score")
            for feature in FEATURES
        ]
        if all(score is not None for score in scores):
            result[prompt_id] = scores
    return result


def method_summary(scores: dict[str, list[float]]) -> dict:
    values = list(scores.values())
    return {
        "n": len(values),
        "joint_ge4": summary.proportion_ci(
            sum(min(row) >= 4 for row in values), len(values)
        ),
        "minimum": summary.mean_ci([min(row) for row in values]),
        "feature_means": {
            feature: round(sum(row[index] for row in values) / len(values), 4)
            for index, feature in enumerate(FEATURES)
        },
    }


def compare(
    left: dict[str, list[float]], right: dict[str, list[float]]
) -> dict[str, dict[str, float | int]]:
    prompt_ids = sorted(left.keys() & right.keys())
    return {
        "joint_ge4": paired_ci(
            [
                int(min(left[prompt_id]) >= 4) - int(min(right[prompt_id]) >= 4)
                for prompt_id in prompt_ids
            ]
        ),
        "minimum": paired_ci(
            [min(left[prompt_id]) - min(right[prompt_id]) for prompt_id in prompt_ids]
        ),
    }


def rank1_composition(
    rows: dict[tuple[str, str, str, str], dict],
) -> dict[str, list[dict]]:
    conditions = []
    all_prompt_ids = {key[1] for key in rows if key[0] == "svd"}
    for mask in range(1, 16):
        answer_id = f"per_r1_{mask:04b}"
        active = summary.active_features("svd", answer_id)
        scored = []
        for prompt_id in all_prompt_ids:
            values = [
                rows.get(("svd", prompt_id, answer_id, feature), {}).get("trait_score")
                for feature in active
            ]
            if all(value is not None for value in values):
                scored.append(values)
        marginal = [
            sum(row[index] >= 4 for row in scored) / len(scored)
            for index in range(len(active))
        ]
        joint = sum(min(row) >= 4 for row in scored) / len(scored)
        conditions.append(
            {
                "condition": answer_id,
                "active_features": list(active),
                "n": len(scored),
                "joint_ge4": round(joint, 4),
                "marginal_ge4": {
                    feature: round(marginal[index], 4)
                    for index, feature in enumerate(active)
                },
                "independence_product": round(math.prod(marginal), 4),
                "mean_minimum": round(sum(min(row) for row in scored) / len(scored), 4),
            }
        )
    return {"conditions": conditions}


def retention(
    rows: dict[tuple[str, str, str, str], dict],
) -> dict[str, dict[str, float | int]]:
    result = {}
    all_id = "per_r1_1111"
    for index, feature in enumerate(FEATURES):
        single_id = f"per_r1_{1 << index:04b}"
        differences = []
        for phase, prompt_id, answer_id, row_feature in rows:
            if phase != "svd" or answer_id != all_id or row_feature != feature:
                continue
            single = rows.get(("svd", prompt_id, single_id, feature))
            if single:
                differences.append(
                    rows[(phase, prompt_id, answer_id, row_feature)]["trait_score"]
                    - single["trait_score"]
                )
        result[feature] = paired_ci(differences)
    return result


def main() -> None:
    args = arguments()
    rows = summary.split_activation_holdout(
        summary.load_results((args.compact_results, args.json_results)),
        args.activation_holdout,
    )
    holdout_ids = {
        json.loads(line)["source_id"]
        for line in args.activation_holdout.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    maps = {
        name: score_map(rows, phase, answer_id, holdout_ids)
        for name, (phase, answer_id) in METHODS.items()
    }
    comparisons = {
        f"{left}_minus_{right}": compare(maps[left], maps[right])
        for left, right in (
            ("gdn_rank4", "gdn_rank1"),
            ("gdn_norm", "gdn_rank1"),
            ("activation_l10_a4", "gdn_rank1"),
            ("activation_l20_a1", "gdn_rank1"),
            ("activation_l10_a4", "gdn_norm"),
            ("activation_l20_a1", "gdn_norm"),
        )
    }
    payload = {
        "schema_version": "1.0",
        "holdout": {
            "methods": {name: method_summary(scores) for name, scores in maps.items()},
            "comparisons": comparisons,
        },
        "rank1_composition": rank1_composition(rows),
        "rank1_four_way_retention": retention(rows),
        "notes": [
            "All holdout comparisons are paired on the same 96 prompts.",
            "Bootstrap intervals resample prompts with a fixed seed.",
            "Scores are discrete Judge ratings; probability-weighted scores were not saved.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
