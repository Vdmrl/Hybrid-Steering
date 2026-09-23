"""Discrimination metrics for Judge runs on pole-labeled data.

`metrics.py` answers "does the Judge reproduce a human label?" on a few dozen
hand-written cases. It cannot answer the question the Judge is actually used
for in steering work: given two answers that differ on the trait, does the
Judge separate them? These metrics answer that, and they need no human score
labels -- only which pole each text was written to express, which large
pole-labeled corpora already carry for free.

Inputs are Judge result rows (as written by the runner) plus a gold sidecar
mapping `prompt_id` -> pole, where pole is "target" or "opposite". For corpora
whose two poles come in aligned pairs (one text rewritten into the other), a
`pair_id` in the sidecar additionally enables the paired statistics, which are
far more sensitive because they cancel per-item difficulty.

Metrics, per feature:
  auc              probability that a random target text outscores a random
                   opposite text, ties counted as half (Mann-Whitney U / n1 n2).
                   0.5 = no discrimination, 1.0 = perfect. Computed on
                   `expected_score` (probability-weighted) and on the integer
                   `trait_score`; the former is the more sensitive endpoint and
                   is why the runner keeps the full distribution.
  mean_gap         mean(target) - mean(opposite) on the integer score.
  cliffs_delta     2 * auc - 1, a rank effect size on the same ordering.
  pair_win_rate    fraction of aligned pairs where target > opposite
  pair_tie_rate    fraction where the two scores are equal
  saturation       share of scores at each end of the scale; a judge pinned at
                   1 or 5 cannot rank anything inside the band it collapsed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TARGET, OPPOSITE = "target", "opposite"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def auc(target_scores: list[float], opposite_scores: list[float]) -> float:
    """Mann-Whitney U / (n1 * n2), ties at half. NaN if either side is empty."""
    if not target_scores or not opposite_scores:
        return float("nan")
    wins = sum(
        1.0 if t > o else 0.5 if t == o else 0.0
        for t in target_scores
        for o in opposite_scores
    )
    return wins / (len(target_scores) * len(opposite_scores))


def split_scores(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    target = [row[key] for row in rows if row["pole"] == TARGET]
    opposite = [row[key] for row in rows if row["pole"] == OPPOSITE]
    return target, opposite


def paired(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("pair_id"):
            by_pair[row["pair_id"]][row["pole"]] = row[key]
    complete = [p for p in by_pair.values() if TARGET in p and OPPOSITE in p]
    if not complete:
        return {"pairs": 0}
    wins = sum(p[TARGET] > p[OPPOSITE] for p in complete)
    ties = sum(p[TARGET] == p[OPPOSITE] for p in complete)
    return {
        "pairs": len(complete),
        "pair_win_rate": wins / len(complete),
        "pair_tie_rate": ties / len(complete),
        "pair_mean_gap": mean([p[TARGET] - p[OPPOSITE] for p in complete]),
    }


def feature_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trait_target, trait_opposite = split_scores(rows, "trait_score")
    exp_target, exp_opposite = split_scores(rows, "expected_score")
    trait_auc = auc(trait_target, trait_opposite)
    counts = Counter(row["trait_score"] for row in rows)
    n = len(rows)
    return {
        "n": n,
        "n_target": len(trait_target),
        "n_opposite": len(trait_opposite),
        "auc": trait_auc,
        "auc_expected_score": auc(exp_target, exp_opposite),
        "cliffs_delta": 2 * trait_auc - 1,
        "mean_target": mean(trait_target),
        "mean_opposite": mean(trait_opposite),
        "mean_gap": mean(trait_target) - mean(trait_opposite),
        "score_distribution": {str(s): counts.get(s, 0) for s in range(1, 6)},
        "saturation_low": counts.get(1, 0) / n if n else float("nan"),
        "saturation_high": counts.get(5, 0) / n if n else float("nan"),
        "paired_trait_score": paired(rows, "trait_score"),
        "paired_expected_score": paired(rows, "expected_score"),
    }


def join_gold(results: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Judge rows + gold sidecar -> flat rows. Judge rows are matched by
    prompt_id; a result with no gold entry is dropped rather than guessed."""
    rows = []
    for result in results:
        entry = gold.get(result["prompt_id"])
        if entry is None:
            continue
        rows.append(
            {
                "prompt_id": result["prompt_id"],
                "feature": result["feature"],
                "pole": entry["pole"],
                "pair_id": entry.get("pair_id"),
                "concept": entry.get("concept", result["feature"]),
                "trait_score": float(result["trait_score"]),
                "expected_score": float(result["score_distribution"]["expected_score"]),
            }
        )
    return rows


def pole_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature[row["feature"]].append(row)
    per_feature = {name: feature_metrics(items) for name, items in sorted(by_feature.items())}
    aucs = [m["auc"] for m in per_feature.values() if m["auc"] == m["auc"]]
    return {
        "n": len(rows),
        "features": len(per_feature),
        "macro_auc": mean(aucs),
        "min_auc": min(aucs) if aucs else float("nan"),
        "by_feature": per_feature,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Judge result rows (JSONL)")
    parser.add_argument("--gold", type=Path, required=True, help="pole sidecar (JSONL)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gold = {row["prompt_id"]: row for row in read_jsonl(args.gold)}
    rows = join_gold(read_jsonl(args.results), gold)
    if not rows:
        raise SystemExit("no Judge result matched the gold sidecar by prompt_id")
    text = json.dumps(pole_metrics(rows), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
