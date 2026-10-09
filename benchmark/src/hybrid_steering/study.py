"""Frozen selection rules and conservative paired inference, without an API or GPU.

The exact interval construction also works when no discordant pairs occur;
unlike a plain percentile bootstrap, it does not collapse to [0, 0].
"""

import itertools
import math
from collections import defaultdict


def condition(row: dict) -> tuple[str, int, float]:
    return row["method"], row.get("layer", -1), float(row["strength"])


def select_triplet(means: dict, methods: list[str], tolerance: float,
                   minimum: float, fixed: dict | None = None) -> list[dict]:
    """Match concept only. Maximise the weakest concept; ties use lower strengths."""
    fixed = fixed or {}
    choices = [[key for key in means if key[0] == method
                and (method not in fixed or key[2] == fixed[method])]
               for method in methods]
    candidates = []
    for keys in itertools.product(*choices):
        values = [means[key] for key in keys]
        if min(values) >= minimum and max(values) - min(values) <= tolerance + 1e-12:
            candidates.append(((-min(values), max(values) - min(values),
                                tuple(key[2] for key in keys)), keys))
    if not candidates:
        raise ValueError("No matched triplet: extend a NEW screening protocol; do not inspect IFEval")
    keys = min(candidates)[1]
    return [{"method": method, "layer": layer, "strength": strength,
             "concept_mean": means[(method, layer, strength)]}
            for method, layer, strength in keys]


def means_by_condition(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[condition(row)].append(row["concept_score"])
    return {key: sum(values) / len(values) for key, values in groups.items()}


def select_levels(points: list[dict], targets: list[float]) -> dict:
    """Match distinct strengths to concept levels, excluding everything after the first peak.

    Scores must already be normalized to 0–1. Selection never reads benchmark
    quality. With fewer eligible strengths than targets, use fewer points and
    explicitly report the unmatched targets.
    """
    points = sorted(points, key=lambda row: row["strength"])
    if (not points or len({row["strength"] for row in points}) != len(points)
            or any(not math.isfinite(row["strength"]) or row["strength"] <= 0
                   or not math.isfinite(row["concept_mean"])
                   or not 0 <= row["concept_mean"] <= 1 for row in points)):
        raise ValueError("Need unique positive strengths and finite concept means in [0, 1]")
    if (not targets or targets != sorted(set(targets))
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in targets)):
        raise ValueError("Targets must be unique, increasing values in [0, 1]")

    peak = max(points, key=lambda row: row["concept_mean"])
    eligible = [row for row in points if row["strength"] <= peak["strength"]]
    count = min(len(eligible), len(targets))
    # At most C(9, 4) candidates for the requested grids; no optimizer dependency.
    candidates = (
        (sum(abs(row["concept_mean"] - target) for row, target in zip(rows, levels)),
         tuple(row["strength"] for row in rows), levels, rows)
        for rows in itertools.combinations(eligible, count)
        for levels in itertools.combinations(targets, count)
    )
    _, _, levels, selected = min(candidates, key=lambda item: item[:3])
    return {
        "peak_strength": peak["strength"], "peak_score": peak["concept_mean"],
        "selected": [{**row, "target": target,
                      "distance": abs(row["concept_mean"] - target)}
                     for row, target in zip(selected, levels)],
        "unmatched_targets": [target for target in targets if target not in levels],
        "discarded_strengths": [row["strength"] for row in points
                                if row["strength"] > peak["strength"]],
    }


def select_nearest_levels(points: list[dict], targets: list[float]) -> dict:
    """Choose one distinct measured strength per ordered target, without peeking at quality."""
    points = sorted(points, key=lambda row: row["strength"])
    if (len(points) < len(targets)
            or len({row["strength"] for row in points}) != len(points)
            or any(not math.isfinite(row["strength"]) or row["strength"] <= 0
                   or not math.isfinite(row["concept_mean"])
                   or not 0 <= row["concept_mean"] <= 1 for row in points)):
        raise ValueError("Need distinct positive strengths with finite rates in [0, 1]")
    if (not targets or targets != sorted(set(targets))
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in targets)):
        raise ValueError("Targets must be distinct increasing values in [0, 1]")
    selected = min(itertools.combinations(points, len(targets)), key=lambda rows: (
        sum(abs(row["concept_mean"] - target) for row, target in zip(rows, targets)),
        tuple(row["strength"] for row in rows)))
    return {"selected": [{**row, "target": target,
                          "distance": abs(row["concept_mean"] - target)}
                         for row, target in zip(selected, targets)],
            "unmatched_targets": [], "discarded_strengths": []}


def binomial_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    return min(1.0, math.fsum(math.exp(
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1)))


def binomial_upper(k: int, n: int, tail: float) -> float:
    """One-sided Clopper–Pearson upper bound, P(Bin(n,p) <= k) = tail."""
    if n < 1 or not 0 <= k <= n or not 0 < tail < 1:
        raise ValueError("Invalid exact binomial interval input")
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(tail) / n)
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if binomial_cdf(k, n, middle) > tail:
            low = middle
        else:
            high = middle
    return high


def paired_interval(a: list[int], b: list[int], maximum: int,
                    alpha: float, two_sided: bool = True) -> tuple[float, float]:
    """Exact conservative bounds for E[A-B] for integer ratings in [0, maximum].

    E[D] = sum_j P(D>=j) - P(D<=-j). Bound each binomial probability
    with Clopper–Pearson, then use a union bound (no independence assumption
    between the events). Independence/exchangeability is across prompt pairs.
    """
    if not a or len(a) != len(b) or any(type(x) not in (int, bool)
            or not 0 <= x <= maximum for x in a + b):
        raise ValueError("Paired complete integer scores are required")
    differences = [x - y for x, y in zip(a, b)]
    n = len(a)
    tail = alpha / ((4 if two_sided else 2) * maximum)
    lower, upper = 0.0, 0.0
    for j in range(1, maximum + 1):
        positive = sum(d >= j for d in differences)
        negative = sum(d <= -j for d in differences)
        lower += 1 - binomial_upper(n - positive, n, tail) - binomial_upper(negative, n, tail)
        upper += binomial_upper(positive, n, tail) - 1 + binomial_upper(n - negative, n, tail)
    return max(-maximum, lower), min(maximum, upper) if two_sided else float(maximum)


def compare(a: list[dict], b: list[dict], *, concept_maximum: int,
            concept_tolerance: float, quality_margin: float, alpha: float) -> dict:
    """A claim needs concept equivalence AND quality non-inferiority on paired prompts."""
    left, right = {r["key"]: r for r in a}, {r["key"]: r for r in b}
    if len(left) != len(a) or len(right) != len(b) or left.keys() != right.keys():
        raise ValueError("Missing or duplicated paired prompt keys")
    keys = sorted(left, key=str)
    if any(left[k]["prompt"] != right[k]["prompt"] for k in keys):
        raise ValueError("Prompts do not match")
    aq, bq = [int(left[k]["ifeval_strict"]) for k in keys], [int(right[k]["ifeval_strict"]) for k in keys]
    ac, bc = [left[k]["concept_score"] for k in keys], [right[k]["concept_score"] for k in keys]
    quality_low, _ = paired_interval(aq, bq, 1, alpha, two_sided=False)
    concept_low, concept_high = paired_interval(ac, bc, concept_maximum, alpha)
    equivalent = concept_low >= -concept_tolerance and concept_high <= concept_tolerance
    noninferior = equivalent and quality_low > -quality_margin
    return {
        "n": len(keys), "alpha_per_contrast": alpha,
        "quality_a": sum(aq) / len(keys), "quality_b": sum(bq) / len(keys),
        "quality_delta": (sum(aq) - sum(bq)) / len(keys), "quality_lower_bound": quality_low,
        "concept_a": sum(ac) / len(keys), "concept_b": sum(bc) / len(keys),
        "concept_delta_interval": [concept_low, concept_high], "concept_equivalent": equivalent,
        "noninferior": noninferior, "superior": equivalent and quality_low > 0,
        "decision": "superior" if equivalent and quality_low > 0 else
                    "noninferior" if noninferior else "not_established",
        "interval_method": "paired ordinal tails + Clopper-Pearson + union bound",
    }
