"""Measured maximin selection with explicit incomplete-coverage fallback."""

import itertools
import math


def select_spread_levels(
    points: list[dict], count: int = 4, preferred_distance: float = 0.2
) -> dict:
    """Maximin selection, preferring the increasing envelope and observed 1.

    Only expression and strength enter the decision. Suggested strengths are
    unmeasured proposals, never selected benchmark conditions.
    """
    if (
        type(count) is not int
        or count < 2
        or not 0 <= preferred_distance <= 1
        or not points
        or len({p["strength"] for p in points}) != len(points)
        or any(
            not math.isfinite(p["strength"])
            or p["strength"] <= 0
            or not math.isfinite(p["expression"])
            or not 0 <= p["expression"] <= 1
            for p in points
        )
    ):
        raise ValueError("Need distinct positive strengths and finite expression in [0, 1]")
    ordered = sorted(points, key=lambda p: p["strength"])
    envelope, best = [], -math.inf
    for point in ordered:
        if point["expression"] > best:
            envelope.append(point)
            best = point["expression"]
    endpoint = next((p for p in envelope if p["expression"] == 1.0), None)
    reasons = []
    if endpoint is None:
        reasons.append("no_observed_expression_one")
    if len(envelope) < count:
        reasons.append("too_few_increasing_points")
    proposals = []
    # Midpoints of large expression gaps: interpolation guides proposals only.
    gaps = sorted(
        zip(envelope, envelope[1:]),
        key=lambda ab: (-(ab[1]["expression"] - ab[0]["expression"]), ab[0]["strength"]),
    )
    for left, right in gaps:
        proposal = (left["strength"] + right["strength"]) / 2
        if proposal not in {p["strength"] for p in ordered}:
            proposals.append(
                {
                    "strength": proposal,
                    "reason": "midpoint_of_expression_gap",
                    "expression_bracket": [left["expression"], right["expression"]],
                    "measured": False,
                }
            )
    if endpoint is None:
        peak = max(ordered, key=lambda p: p["expression"])
        if peak["strength"] == ordered[-1]["strength"]:
            proposals += [
                {
                    "strength": ordered[-1]["strength"] * factor,
                    "reason": "explore_saturation_no_guarantee",
                    "measured": False,
                }
                for factor in (1.25, 1.5)
            ]
        else:
            proposals.append(
                {
                    "strength": (peak["strength"] + ordered[ordered.index(peak) + 1]["strength"])
                    / 2,
                    "reason": "refine_near_peak_before_degradation",
                    "measured": False,
                }
            )
    result = {
        "status": "selected_with_incomplete_coverage" if reasons else "selected",
        "reasons": reasons,
        "selected": [],
        "envelope": envelope,
        "discarded_strengths": [p["strength"] for p in ordered if p not in envelope],
        "minimum_expression_distance": None,
        "preferred_distance": preferred_distance,
        "preferred_distance_met": False,
        "suggested_points": proposals,
    }
    # Incomplete concept coverage must not exclude a series from IFEval.
    # Fall back to measured points; never invent expression=1 or new strengths.
    eligible = envelope if len(envelope) >= count else ordered
    if endpoint is None:
        endpoint = max(ordered, key=lambda p: (p["expression"], -p["strength"]))
    actual_count = min(count, len(eligible))
    candidates = []
    for subset in itertools.combinations([p for p in eligible if p != endpoint], actual_count - 1):
        chosen = sorted([*subset, endpoint], key=lambda p: p["strength"])
        expressions = sorted(p["expression"] for p in chosen)
        distance = min((b - a for a, b in zip(expressions, expressions[1:])), default=0.0)
        candidates.append(
            ((-round(distance, 12), tuple(p["strength"] for p in chosen)), chosen, distance)
        )
    _, selected, distance = min(candidates, key=lambda candidate: candidate[0])
    enough = distance + 1e-12 >= preferred_distance
    result.update(
        selected=selected,
        minimum_expression_distance=distance,
        preferred_distance_met=enough,
        status="selected_with_incomplete_coverage"
        if reasons
        else "selected"
        if enough
        else "selected_with_small_gap",
        observed_expression_one=any(p["expression"] == 1 for p in selected),
        used_non_envelope_points=any(p not in envelope for p in selected),
        selected_count=actual_count,
        suggested_points=[] if enough and not reasons else proposals,
    )
    return result
