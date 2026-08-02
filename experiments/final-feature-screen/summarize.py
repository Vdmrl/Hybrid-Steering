"""Summarize blinded factorial judgments without copying raw artifacts into Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean_ci(values: list[float], label: str, samples: int = 2_000) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(
        fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)
    )
    return {
        "n": n,
        "mean": round(fmean(values), 6),
        "ci95": [
            round(boots[int(samples * 0.025)], 6),
            round(boots[int(samples * 0.975)], 6),
        ],
    }


def condition_key(row: dict[str, Any]) -> str:
    return row["condition"]


def result_key(result: dict[str, Any]) -> str:
    suffix = f":{result['feature']}"
    prompt_id = result["prompt_id"]
    if not prompt_id.endswith(suffix):
        raise ValueError(f"unexpected judge prompt id: {prompt_id}")
    return prompt_id[: -len(suffix)]


def score(result: dict[str, Any], name: str) -> float:
    if name == "expected_score":
        return float(result["score_distribution"]["expected_score"])
    if name == "p_ge4":
        probabilities = result["score_distribution"]["probabilities"]
        return float(probabilities.get("4", probabilities.get(4, 0))) + float(
            probabilities.get("5", probabilities.get(5, 0))
        )
    if name == "trait_score":
        return float(result["trait_score"])
    raise ValueError(name)


def metric_summary(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, dict[str, Any]]],
    feature: str,
) -> dict[str, Any]:
    available = [row for row in rows if row["task_id"] in results.get(feature, {})]
    return {
        metric: mean_ci(
            [score(results[feature][row["task_id"]], metric) for row in available],
            f"{feature}:{metric}:{condition_key(available[0]) if available else 'missing'}",
        )
        for metric in ("trait_score", "expected_score", "p_ge4")
    }


def paired_delta(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    results: dict[str, dict[str, dict[str, Any]]],
    feature: str,
    metric: str,
    label: str,
) -> dict[str, Any]:
    left_by_prompt = {row["prompt_id"]: row for row in left}
    right_by_prompt = {row["prompt_id"]: row for row in right}
    values = []
    for prompt_id in sorted(left_by_prompt.keys() & right_by_prompt.keys()):
        lhs = results.get(feature, {}).get(left_by_prompt[prompt_id]["task_id"])
        rhs = results.get(feature, {}).get(right_by_prompt[prompt_id]["task_id"])
        if lhs and rhs:
            values.append(score(lhs, metric) - score(rhs, metric))
    return mean_ci(values, label)


def load_results(
    judge_dir: Path, features: list[str]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, int]]:
    values: dict[str, dict[str, dict[str, Any]]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    for feature in [*features, "answer_quality"]:
        path = judge_dir / f"{feature}.judgments.jsonl"
        if not path.exists():
            continue
        values[feature] = {}
        for result in read_jsonl(path):
            values[feature][result_key(result)] = result
            item = result.get("provenance", {}).get("usage", {})
            for key in usage:
                usage[key] += int(item.get(key, 0) or 0)
    return values, usage


def summarize(generations: Path, judge_dir: Path, selection: Path) -> dict[str, Any]:
    selected = json.loads(selection.read_text(encoding="utf-8"))
    features = [str(row["name"]) for row in selected["features"]]
    rows = read_jsonl(generations)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[condition_key(row)].append(row)
    results, usage = load_results(judge_dir, features)
    missing_features = [
        name for name in [*features, "answer_quality"] if name not in results
    ]
    baseline = by_condition.get("baseline", [])
    conditions: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        active = (
            list(condition_rows[0].get("active_features", [])) if condition_rows else []
        )
        feature_values = {
            feature: metric_summary(condition_rows, results, feature)
            for feature in features
            if feature in results
        }
        quality = (
            metric_summary(condition_rows, results, "answer_quality")
            if "answer_quality" in results
            else None
        )
        deltas = {
            feature: {
                metric: paired_delta(
                    condition_rows,
                    baseline,
                    results,
                    feature,
                    metric,
                    f"baseline:{condition}:{feature}:{metric}",
                )
                for metric in ("trait_score", "expected_score", "p_ge4")
            }
            for feature in features
            if feature in results
        }
        quality_delta = (
            paired_delta(
                condition_rows,
                baseline,
                results,
                "answer_quality",
                "expected_score",
                f"baseline:{condition}:quality",
            )
            if "answer_quality" in results
            else None
        )
        joint = []
        for row in condition_rows:
            if not active or any(
                row["task_id"] not in results.get(feature, {}) for feature in active
            ):
                continue
            joint.append(
                float(
                    all(
                        results[feature][row["task_id"]]["trait_score"] >= 4
                        for feature in active
                    )
                )
            )
        conditions[condition] = {
            "active_features": active,
            "n_trait": len(condition_rows),
            "feature_metrics": feature_values,
            "delta_vs_baseline": deltas,
            "quality": quality,
            "quality_delta_vs_baseline": quality_delta,
            "joint_success": mean_ci(joint, f"joint:{condition}"),
        }

    leakage: dict[str, dict[str, Any]] = {}
    for added in features:
        leakage[added] = {}
        comparisons: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        for condition, condition_rows in by_condition.items():
            active = set(condition_rows[0].get("active_features", []))
            if added not in active:
                continue
            base = "+".join(
                feature for feature in features if feature in active - {added}
            )
            base = base or "baseline"
            if base in by_condition:
                comparisons.append((condition_rows, by_condition[base]))
        for evaluated in features:
            values = []
            for left, right in comparisons:
                left_by_prompt = {row["prompt_id"]: row for row in left}
                right_by_prompt = {row["prompt_id"]: row for row in right}
                for prompt_id in left_by_prompt.keys() & right_by_prompt.keys():
                    lhs = results.get(evaluated, {}).get(
                        left_by_prompt[prompt_id]["task_id"]
                    )
                    rhs = results.get(evaluated, {}).get(
                        right_by_prompt[prompt_id]["task_id"]
                    )
                    if lhs and rhs:
                        values.append(
                            score(lhs, "expected_score") - score(rhs, "expected_score")
                        )
            leakage[added][evaluated] = mean_ci(values, f"leakage:{added}:{evaluated}")

    full_condition = "+".join(features)
    retention = {}
    if full_condition in by_condition:
        for feature in features:
            if feature in by_condition:
                retention[feature] = paired_delta(
                    by_condition[full_condition],
                    by_condition[feature],
                    results,
                    feature,
                    "expected_score",
                    f"retention:{feature}",
                )
    pairs = {
        condition: conditions[condition]
        for condition in conditions
        if len(conditions[condition]["active_features"]) == 2
    }
    return {
        "schema_version": "final-feature-screen-summary-1",
        "features": features,
        "selection": selected,
        "n_generation_rows": len(rows),
        "n_prompts": len(baseline),
        "missing_judge_features": missing_features,
        "judge_usage": usage,
        "conditions": conditions,
        "pair_conditions": pairs,
        "leakage_matrix_expected_score": leakage,
        "singleton_retention_expected_score": retention,
        "full_condition": full_condition if full_condition in conditions else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.generations, args.judge_dir, args.selection)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
