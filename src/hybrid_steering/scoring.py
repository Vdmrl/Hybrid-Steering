"""Concept hit, quality, and scale choice shared by the steering sweeps.

Scale 1 injects a direction whose Frobenius norm equals the natural GDN state
norm at the last prompt token. ``choose`` keeps, per method, the smallest
scale among those with the highest concept rate whose quality and repetition
stay near the unsteered baseline.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

from .cache import gdn_layers
from .detect import is_language
from .extract import final_states
from .judge import score_rows, score_steering
from .runner import Runner
from .runtime import chat_prompts
from .state import truncate_direction

METHODS = {"rank1": 1, "rank2": 2, "full": None, "clamp": 1}
QUALITY_DROP = 0.5
REPETITION_RISE = 0.1


def natural_norm(model, tokenizer, questions: list[str]) -> float:
    runner = Runner(model, tokenizer, gdn_layers(model), normalize=False)
    states = final_states(runner, chat_prompts(tokenizer, questions))
    squares = sum(state.float().pow(2).sum((1, 2, 3)) for state in states.values())
    return float(squares.sqrt().mean())


def injected_norm(direction: dict, rank: int | None) -> float:
    truncated = truncate_direction(direction, rank)
    return math.sqrt(sum(float(tensor.pow(2).sum()) for tensor in truncated.values()))


def repetition(token_ids: list[int], n: int = 4) -> float:
    grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def score(
    rows: list[dict], feature: str, batch_size: int, *, settings_path: Path | None = None
) -> None:
    language = is_language(feature)
    score_rows(
        rows,
        feature,
        prompt_field="question",
        batch_size=batch_size,
        settings_path=settings_path,
    )
    if language:
        judgments = score_steering(
            [(row["question"], row["response"]) for row in rows],
            "answer_quality",
            batch_size=batch_size,
            settings_path=settings_path,
        )
        for row, judgment in zip(rows, judgments, strict=True):
            row["content_quality"] = judgment.content_quality if judgment else None
            row["evaluable"] = judgment.evaluable if judgment else None
            row["flags"] = list(judgment.flags) if judgment else []
    for row in rows:
        if language:
            row["hit"] = int(row["concept_score"] >= 1)
        elif row.get("evaluable") is False:
            row["hit"] = 0
        elif row.get("concept_score") is None:
            row["hit"] = None
        else:
            row["hit"] = int(row["concept_score"] >= 2)


def summarize(rows: list[dict]) -> list[dict]:
    cells: dict[tuple, list] = {}
    for row in rows:
        cells.setdefault((row["method"], row["scale"]), []).append(row)
    summary = []
    for (method, scale), items in sorted(cells.items(), key=lambda item: (item[0][0], item[0][1])):
        hits = [row["hit"] for row in items if row["hit"] is not None]
        quality = [
            row["content_quality"] for row in items if row.get("content_quality") is not None
        ]
        low, high = wilson(sum(hits), len(hits))
        summary.append(
            {
                "method": method,
                "scale": scale,
                "n": len(hits),
                "concept_rate": sum(hits) / len(hits) if hits else math.nan,
                "ci_low": low,
                "ci_high": high,
                "quality": sum(quality) / len(quality) if quality else math.nan,
                "repetition": sum(row["repetition"] for row in items) / len(items),
                "unevaluable": sum(row.get("evaluable") is False for row in items) / len(items),
            }
        )
    return summary


def choose(summary: list[dict], methods: dict = METHODS) -> dict[str, dict]:
    baseline = next(item for item in summary if item["method"] == "baseline")
    chosen = {}
    for method in methods:
        candidates = [
            item
            for item in summary
            if item["method"] == method
            and item["quality"] >= baseline["quality"] - QUALITY_DROP
            and item["repetition"] <= baseline["repetition"] + REPETITION_RISE
        ]
        if candidates:
            chosen[method] = max(
                candidates, key=lambda item: (item["concept_rate"], -item["scale"])
            )
    return chosen


def select_pareto_scales(
    cells: list[dict], targets: tuple[float, ...] = (0.3, 0.5, 0.7, 0.99)
) -> list[dict]:
    """Choose distinct positive pre-peak concept levels by absolute target error.

    Each cell has ``scale`` and a mean ``concept_rate`` already normalized to
    0–1. Quality is deliberately absent: the full benchmarks measure it.
    """
    if len(targets) != 4 or tuple(sorted(set(targets))) != targets:
        raise ValueError("need four increasing concept targets")
    ordered = sorted(cells, key=lambda row: row["scale"])
    if len({row["scale"] for row in ordered}) != len(ordered) or any(
        not math.isfinite(row["scale"])
        or row["scale"] <= 0
        or not math.isfinite(row["concept_rate"])
        or not 0 <= row["concept_rate"] <= 1
        for row in ordered
    ):
        raise ValueError("need distinct positive scales and concept rates in [0, 1]")
    if not ordered:
        raise ValueError("no sweep cells")
    peak = max(ordered, key=lambda row: row["concept_rate"])["scale"]
    eligible = sorted(
        (row for row in ordered if row["scale"] <= peak and row["concept_rate"] > 0),
        key=lambda row: (row["concept_rate"], row["scale"]),
    )
    if len({row["concept_rate"] for row in eligible}) < 4:
        raise ValueError("fewer than four distinct nonzero concept levels before the first peak")

    def criterion(rows):
        rates = [row["concept_rate"] for row in rows]
        return (
            sum(abs(rate - target) for rate, target in zip(rates, targets, strict=True)),
            -min(right - left for left, right in itertools.pairwise(rates)),
            tuple(row["scale"] for row in rows),
        )

    candidates = (
        rows
        for rows in itertools.combinations(eligible, 4)
        if len({row["concept_rate"] for row in rows}) == 4
    )
    selected = min(candidates, key=criterion)
    return [
        {**row, "target": target, "distance": abs(row["concept_rate"] - target)}
        for row, target in zip(selected, targets, strict=True)
    ]
