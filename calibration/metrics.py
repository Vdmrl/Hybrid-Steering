"""Compute compact, deterministic metrics for Judge calibration outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCORES = range(1, 6)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def confusion(rows: list[dict[str, Any]]) -> list[list[int]]:
    return [
        [
            sum(
                row["expected_score"] == expected and row["trait_score"] == predicted
                for row in rows
            )
            for predicted in SCORES
        ]
        for expected in SCORES
    ]


def weighted_kappa(rows: list[dict[str, Any]]) -> float:
    n = len(rows)
    if not n:
        return float("nan")
    observed = (
        sum(
            ((predicted - expected) / 4) ** 2
            for row in rows
            for expected, predicted in [(row["expected_score"], row["trait_score"])]
        )
        / n
    )
    expected_counts = Counter(row["expected_score"] for row in rows)
    predicted_counts = Counter(row["trait_score"] for row in rows)
    expected_disagreement = sum(
        expected_counts[e] * predicted_counts[p] * ((p - e) / 4) ** 2
        for e in SCORES
        for p in SCORES
    ) / (n * n)
    return (
        1.0
        if expected_disagreement == 0 and observed == 0
        else 1 - observed / expected_disagreement
    )


def brier(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        probs = {
            int(k): float(v)
            for k, v in row["score_distribution"]["probabilities"].items()
        }
        values.append(
            sum((probs[s] - (s == row["expected_score"])) ** 2 for s in SCORES)
        )
    return mean(values)


def logloss(rows: list[dict[str, Any]], floor: float = 1e-12) -> float:
    return mean(
        [
            -math.log(
                max(
                    float(
                        row["score_distribution"]["probabilities"].get(
                            str(row["expected_score"]), 0
                        )
                    ),
                    floor,
                )
            )
            for row in rows
        ]
    )


def per_score(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for score in SCORES:
        actual = sum(row["expected_score"] == score for row in rows)
        predicted = sum(row["trait_score"] == score for row in rows)
        true_positive = sum(
            row["expected_score"] == score and row["trait_score"] == score
            for row in rows
        )
        result[str(score)] = {
            "support": actual,
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / actual if actual else 0.0,
        }
    return result


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    absolute_errors = [abs(row["trait_score"] - row["expected_score"]) for row in rows]
    expected_errors = [
        abs(row["score_distribution"]["expected_score"] - row["expected_score"])
        for row in rows
    ]
    macro_precision = mean([item["precision"] for item in per_score(rows).values()])
    macro_recall = mean([item["recall"] for item in per_score(rows).values()])
    if macro_precision + macro_recall:
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    else:
        macro_f1 = 0.0
    high_confidence = [
        row
        for row in rows
        if row["trait_score"] != row["expected_score"]
        and row["score_distribution"]["chosen_score_probability"] >= 0.8
    ]
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature[row["feature"]].append(row)
    return {
        "n": len(rows),
        "exact_accuracy": sum(error == 0 for error in absolute_errors) / len(rows),
        "mae": mean([float(error) for error in absolute_errors]),
        "errors_gt_one": sum(error > 1 for error in absolute_errors) / len(rows),
        "confusion": confusion(rows),
        "weighted_kappa": weighted_kappa(rows),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_score": per_score(rows),
        "expected_score_mae": mean(expected_errors),
        "brier": brier(rows),
        "logloss": logloss(rows),
        "mean_entropy": mean([row["score_distribution"]["entropy"] for row in rows]),
        "high_confidence_errors": len(high_confidence),
        "expected_distribution": dict(
            sorted(Counter(row["expected_score"] for row in rows).items())
        ),
        "predicted_distribution": dict(
            sorted(Counter(row["trait_score"] for row in rows).items())
        ),
        "by_feature": {
            feature: metrics(feature_rows)
            for feature, feature_rows in sorted(by_feature.items())
        }
        if len(by_feature) > 1
        else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = metrics(rows)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
