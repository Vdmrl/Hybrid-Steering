"""Concept-versus-filler-length curves for the computed forgetting runs.

One plot per concept shows baseline, full rank, residual, and clamp. A second
shows baseline, the three ranks, and the same ranks with frozen attention.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {
    "baseline": "#475569",
    "rank1": "#d62728",
    "rank2": "#9467bd",
    "full": "#15803d",
    "clamp": "#ea580c",
    "all": "#2563eb",
}
LABELS = {
    "en-ru": "Russian",
    "en-fr": "French",
    "en-zh": "Chinese",
    "en-ar": "Arabic",
    "fairytale": "fairytale",
    "numbered": "numbered",
    "plain-theistic_framing": "theistic",
    "probabilistic": "probabilistic",
    "optimistic/fairytale": "fairytale, residual scale 0.5",
    "optimistic/theistic": "theistic, residual scale 1",
}
NAMES = {
    "baseline": "baseline",
    "rank1": "rank 1",
    "rank2": "rank 2",
    "full": "full",
    "clamp": "clamp",
    "all": "residual",
}
RANK_PARTS = ("normal", "short", "long")
FROZEN_PARTS = ("frozen", "frozen-short", "frozen-long")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def concept_name(slug: str) -> str:
    return LABELS.get(slug, slug)


def file_slug(slug: str) -> str:
    names = {
        "en-ru": "russian",
        "en-fr": "french",
        "en-zh": "chinese",
        "en-ar": "arabic",
        "plain-theistic_framing": "theistic",
        "optimistic/fairytale": "fairytale-residual-scale-0.5",
        "optimistic/theistic": "theistic-residual-scale-1",
    }
    return names.get(slug, slug.replace("/", "-"))


def chosen_scales(summary: dict) -> dict[str, float]:
    grid = summary.get("grid") or {}
    scales = {method: float(values[0]) for method, values in grid.items() if values}
    if "scales" in summary and isinstance(summary["scales"], dict):
        scales.update({method: float(scale) for method, scale in summary["scales"].items()})
    scales.setdefault("baseline", 0.0)
    return scales


def score_of(row: dict) -> float | None:
    if row.get("concept_score") is not None:
        return float(row["concept_score"])
    if row.get("hit") is not None:
        return float(row["hit"])
    return None


def bootstrap(pairs: list[tuple[int, float]], seed: int) -> tuple[float, float, float]:
    by_question: dict[int, list[float]] = defaultdict(list)
    for index, value in pairs:
        by_question[index].append(value)
    keys = list(by_question)

    def mean(sample: list[int]) -> float:
        values = [value for key in sample for value in by_question[key]]
        return sum(values) / len(values)

    point = mean(keys)
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(keys, k=len(keys))) for _ in range(1000))
    return point, draws[25], draws[974]


def from_cells(cells: list[dict]) -> tuple[float, float, float]:
    total = sum(cell["n"] for cell in cells)
    point = sum(cell["n"] * cell["concept_rate"] for cell in cells) / total
    variance = 0.0
    for cell in cells:
        width = cell.get("ci_high")
        if width is None or cell.get("ci_low") is None:
            continue
        error = (cell["ci_high"] - cell["ci_low"]) / (2 * 1.96)
        variance += (cell["n"] / total) ** 2 * error**2
    if variance == 0:
        return point, point, point
    margin = 1.96 * math.sqrt(variance)
    return point, point - margin, point + margin


def series_for(
    root: Path, parts: tuple[str, ...], scales: dict[str, float]
) -> dict[str, list[dict]]:
    rows_at: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    cells_at: dict[tuple, list[dict]] = defaultdict(list)
    for part in parts:
        directory = root / part
        for row in read_jsonl(directory / "rows.jsonl"):
            value = score_of(row)
            if value is None or scales.get(row["method"]) != float(row["scale"]):
                continue
            rows_at[(row["method"], row["length"])].append((row["index"], value))
        for cell in load_summary(directory / "summary.json").get("cells") or []:
            if scales.get(cell["method"]) != float(cell["scale"]):
                continue
            cells_at[(cell["method"], cell["length"])].append(cell)
    series: dict[str, list[dict]] = {}
    methods = {method for method, _length in rows_at} | {method for method, _length in cells_at}
    for method in methods:
        lengths = sorted(
            {length for (name, length) in rows_at if name == method}
            | {length for (name, length) in cells_at if name == method}
        )
        points = []
        for length in lengths:
            if rows_at[(method, length)]:
                point, low, high = bootstrap(
                    rows_at[(method, length)],
                    seed=zlib.crc32(f"{method}:{length}".encode()),
                )
            else:
                point, low, high = from_cells(cells_at[(method, length)])
            points.append(
                {
                    "length": length,
                    "scale": scales[method],
                    "score": point,
                    "low": low,
                    "high": high,
                }
            )
        series[method] = points
    return series


def _nice_step(span: float) -> float:
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if span / step <= 6:
            return step
    return 1.0


def _y_limits(peak: float) -> tuple[float, float, float]:
    """Top of the axis sits on the data. Zero is kept just above the frame."""
    step = _nice_step(peak if peak > 0 else 1.0)
    y_max = max(step, math.ceil((peak - 1e-9) / step) * step)
    return -0.04 * y_max, y_max, step


def _ticks(y_max: float, step: float) -> list[float]:
    ticks = []
    tick = 0.0
    while tick <= y_max + 1e-9:
        ticks.append(round(tick, 10))
        tick += step
    return ticks


def draw(title: str, y_title: str, lines: list[dict], stem: Path) -> None:
    plt.switch_backend("Agg")
    lengths = sorted({point["length"] for line in lines for point in line["points"]})
    peak = max(point["score"] for line in lines for point in line["points"])
    y_min, y_max, step = _y_limits(peak)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for line in lines:
        points = line["points"]
        xs = [point["length"] + 1 for point in points]
        ys = [point["score"] for point in points]
        lower = [max(0.0, point["score"] - point["low"]) for point in points]
        upper = [max(0.0, point["high"] - point["score"]) for point in points]
        ax.errorbar(
            xs,
            ys,
            yerr=[lower, upper],
            color=line["color"],
            linestyle="--" if line.get("dash") else "-",
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            capsize=2,
            label=line["label"],
        )
    if len(lengths) > 1:
        ax.set_xscale("log")
    ax.set_xticks([length + 1 for length in lengths])
    ax.set_xticklabels([f"{length:g}" for length in lengths])
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(_ticks(y_max, step))
    ax.set_xlabel("filler tokens")
    ax.set_ylabel(y_title)
    ax.set_title(title)
    ax.grid(True, color="#e7edf4")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=200)
    plt.close(fig)


def y_title(lines: list[dict]) -> str:
    peak = max(point["score"] for line in lines for point in line["points"])
    return "concept score" if peak > 1 else "concept rate"


def _line(method: str, points: list[dict], *, frozen: bool = False) -> dict:
    return {
        "key": method,
        "label": f"{NAMES[method]}, frozen" if frozen else NAMES[method],
        "color": COLORS[method],
        "dash": "7 4" if frozen else "",
        "points": points,
    }


def _write_plot(output: Path, slug: str, kind: str, lines: list[dict], curves: list[dict]) -> None:
    kept = [line for line in lines if line["points"]]
    if not any(line["key"] != "baseline" for line in kept):
        return
    name = concept_name(slug)
    if slug.startswith("optimistic/"):
        destination = output / "higher-scale"
        stem = file_slug(slug)
    else:
        destination = output
        stem = (
            f"forgetting-{file_slug(slug)}"
            if kind == "compare"
            else f"forgetting-{file_slug(slug)}-ranks"
        )
    title = name if kind == "compare" else f"{name}, ranks"
    destination.mkdir(parents=True, exist_ok=True)
    draw(title, y_title(kept), kept, destination / stem)
    for line in kept:
        for point in line["points"]:
            curves.append(
                {
                    "concept": name,
                    "setting": kind,
                    "method": line["label"],
                    "length": point["length"],
                    "score": point["score"],
                    "low": point["low"],
                    "high": point["high"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forgetting", type=Path, default=Path("runs/forgetting"))
    parser.add_argument("--residual", type=Path, default=Path("runs/forgetting-residual"))
    parser.add_argument("--output", type=Path, default=Path("runs/results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for stale in (*args.output.glob("forgetting-*.svg"), *args.output.glob("forgetting-*.png")):
        stale.unlink()
    curves: list[dict] = []
    residuals: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(args.residual.rglob("lengths")):
        slug = str(path.relative_to(args.residual).parent)
        summary = load_summary(path / "summary.json")
        residuals[slug] = series_for(path.parent, ("lengths",), chosen_scales(summary))
    concepts = [
        path.name
        for path in sorted(args.forgetting.iterdir())
        if path.is_dir() and (path / "normal").is_dir()
    ]
    for slug in [*concepts, *(slug for slug in residuals if slug not in concepts)]:
        root = args.forgetting / slug
        ranks: dict[str, list[dict]] = {}
        frozen: dict[str, list[dict]] = {}
        if (root / "normal").is_dir():
            scales = chosen_scales(load_summary(root / "normal" / "summary.json"))
            ranks = series_for(root, RANK_PARTS, scales)
            frozen_summary = load_summary(root / "frozen" / "summary.json")
            if frozen_summary:
                frozen = series_for(root, FROZEN_PARTS, chosen_scales(frozen_summary))
        residual = residuals.get(slug, {})
        baseline = ranks.get("baseline") or residual.get("baseline")
        compare = []
        if baseline:
            compare.append(_line("baseline", baseline))
        if ranks.get("full"):
            compare.append(_line("full", ranks["full"]))
        if residual.get("all"):
            compare.append(_line("all", residual["all"]))
        if ranks.get("clamp"):
            compare.append(_line("clamp", ranks["clamp"]))
        _write_plot(args.output, slug, "compare", compare, curves)
        rank_lines = []
        if ranks.get("baseline"):
            rank_lines.append(_line("baseline", ranks["baseline"]))
        for method in ("rank1", "rank2", "full"):
            if ranks.get(method):
                rank_lines.append(_line(method, ranks[method]))
        for method in ("rank1", "rank2", "full"):
            if frozen.get(method):
                rank_lines.append(_line(method, frozen[method], frozen=True))
        _write_plot(args.output, slug, "ranks", rank_lines, curves)
    (args.output / "forgetting-curves.json").write_text(
        json.dumps(curves, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(list(args.output.glob('forgetting-*.png')))} plots to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
