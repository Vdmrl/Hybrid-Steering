"""Paper and appendix figures for the language forgetting curves.

The article figure is one panel: the one-shot full direction for Russian,
French, and Arabic. Chinese, the rank curves, and the other concepts are
appendix figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

CURVES = Path("runs/results/forgetting-curves.json")
OUTPUT = Path("runs/results/paper-ready")

# Teal and burnt orange from the trait-strength figure, plus two neighbors.
TEAL = "#1B7F72"
ORANGE = "#D4723A"
SLATE = "#3D5A80"
SAND = "#C4A35A"
GRAY = "#8E939A"
INK = "#2C2C2C"

MAIN = ("Russian", "French", "Arabic")
LANGUAGES = ("Russian", "French", "Chinese", "Arabic")
LANGUAGE_COLOR = {
    "Russian": ORANGE,
    "French": TEAL,
    "Chinese": SLATE,
    "Arabic": SAND,
}


def load() -> list[dict]:
    return json.loads(CURVES.read_text(encoding="utf-8"))


def points(curves: list[dict], concept: str, setting: str, method: str) -> list[dict]:
    rows = [
        row
        for row in curves
        if row["concept"] == concept and row["setting"] == setting and row["method"] == method
    ]
    return sorted(rows, key=lambda row: row["length"])


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#4A4A4A",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def frame(ax, ylabel: str, y_max: float) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("filler tokens", labelpad=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03 * y_max, y_max * 1.04)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.5)


def ticks(ax, rows: list[dict], *, rotate: bool = False) -> None:
    lengths = sorted({int(row["length"]) for row in rows})
    ax.set_xlim(0.7, (max(lengths) + 1) * 1.4)
    ax.set_xticks([length + 1 for length in lengths])
    ax.set_xticklabels(
        [str(length) for length in lengths],
        rotation=30 if rotate else 0,
        ha="right" if rotate else "center",
        rotation_mode="anchor",
    )
    ax.minorticks_off()


def band(
    ax,
    rows: list[dict],
    color: str,
    *,
    ls: str = "-",
    lw: float = 1.8,
    alpha: float = 0.18,
    label: str | None = None,
    marker: str = "o",
    z: int = 2,
) -> None:
    if not rows:
        return
    xs = [row["length"] + 1 for row in rows]
    ys = [row["score"] for row in rows]
    lo = [max(0.0, row["low"]) for row in rows]
    hi = [row["high"] for row in rows]
    if alpha > 0:
        ax.fill_between(xs, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=z)
    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=ls,
        linewidth=lw,
        marker=marker,
        markersize=5.5,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=label,
        zorder=z + 1,
        solid_capstyle="round",
    )


def ceiling(rows: list[dict]) -> tuple[float, str]:
    peak = max((max(row["score"], row["high"]) for row in rows), default=1.0)
    if peak <= 1:
        return 1.0, "concept rate"
    top = 1
    while top < peak:
        top += 1
    return float(top), "concept score"


def across(
    curves: list[dict], setting: str, method: str, languages: tuple[str, ...] = MAIN
) -> list[dict]:
    buckets: dict[float, list[dict]] = {}
    for language in languages:
        for row in points(curves, language, setting, method):
            buckets.setdefault(row["length"], []).append(row)
    averaged = []
    for length, rows in sorted(buckets.items()):
        averaged.append(
            {
                "length": length,
                "score": sum(row["score"] for row in rows) / len(rows),
                "low": sum(row["low"] for row in rows) / len(rows),
                "high": sum(row["high"] for row in rows) / len(rows),
            }
        )
    return averaged


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main_figure(curves: list[dict], path: Path) -> None:
    """One-shot full direction. Chinese stays in the appendix."""
    style()
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    shown: list[dict] = []
    for language in MAIN:
        rows = points(curves, language, "compare", "full")
        shown.extend(rows)
        band(ax, rows, LANGUAGE_COLOR[language], label=language, z=3)
    for method, color, ls, label, marker in (
        ("baseline", GRAY, "-", "baseline", "o"),
        ("residual", INK, "-", "residual", "o"),
        ("clamp", INK, (0, (3.2, 1.5)), "clamp", "s"),
    ):
        rows = across(curves, "compare", method)
        shown.extend(rows)
        band(ax, rows, color, ls=ls, lw=1.45, alpha=0.12, label=label, marker=marker, z=4)
    frame(ax, "concept rate", 1.0)
    ticks(ax, shown)
    ax.legend(loc="center left", frameon=False, handlelength=2.4, borderaxespad=0.6)
    fig.tight_layout()
    _save(fig, path)


def grid_figure(curves: list[dict], path: Path, methods: list[tuple[str, str, str]]) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True, sharey=True)
    shown: list[dict] = []
    for ax, language in zip(axes.ravel(), LANGUAGES, strict=True):
        for method, color, label in methods:
            rows = points(curves, language, "ranks", method)
            shown.extend(rows)
            band(ax, rows, color, label=label if language == "Russian" else None)
        ax.set_title(language, loc="left", pad=6)
        frame(ax, "concept rate" if language in ("Russian", "Chinese") else "", 1.0)
        if language in ("Russian", "French"):
            ax.set_xlabel("")
        ticks(ax, shown, rotate=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, path)


def chinese_figure(curves: list[dict], path: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    shown: list[dict] = []
    for method, color, ls, label, marker in (
        ("baseline", GRAY, "-", "baseline", "o"),
        ("full", ORANGE, "-", "full", "o"),
        ("residual", TEAL, "-", "residual", "o"),
        ("clamp", SLATE, (0, (3.2, 1.5)), "clamp", "s"),
    ):
        rows = points(curves, "Chinese", "compare", method)
        shown.extend(rows)
        band(ax, rows, color, ls=ls, label=label, marker=marker)
    frame(ax, "concept rate", 1.0)
    ticks(ax, shown)
    ax.set_title("Chinese", loc="left", pad=8)
    ax.legend(loc="center left", frameon=False, handlelength=2.4)
    fig.tight_layout()
    _save(fig, path)


def concept_figure(curves: list[dict], concept: str, path: Path) -> None:
    compare_methods = [
        ("baseline", GRAY, "-", "baseline"),
        ("full", ORANGE, "-", "full"),
        ("residual", TEAL, "-", "residual"),
        ("clamp", SLATE, (0, (3, 1.4)), "clamp"),
    ]
    rank_methods = [
        ("baseline", GRAY, "-", "baseline"),
        ("rank 1", TEAL, "-", "rank 1"),
        ("rank 2", SAND, "-", "rank 2"),
        ("full", ORANGE, "-", "full"),
    ]
    compare = [
        (method, color, ls, label)
        for method, color, ls, label in compare_methods
        if points(curves, concept, "compare", method)
    ]
    ranks = [
        (method, color, ls, label)
        for method, color, ls, label in rank_methods
        if points(curves, concept, "ranks", method)
    ]
    panels = []
    if compare:
        panels.append(("One-shot write", "compare", compare))
    if len(ranks) > 1:
        panels.append(("Ranks", "ranks", ranks))
    if not panels:
        return
    style()
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.3 * len(panels), 3.9), sharey=True, squeeze=False
    )
    gathered: list[dict] = []
    for ax, (title, setting, methods) in zip(axes[0], panels, strict=True):
        for method, color, ls, label in methods:
            rows = points(curves, concept, setting, method)
            gathered.extend(rows)
            band(ax, rows, color, ls=ls, label=label, marker="o" if ls == "-" else "s")
        y_max, ylabel = ceiling(gathered)
        frame(ax, ylabel, y_max)
        ticks(ax, gathered, rotate=True)
        ax.set_title(title, loc="left", pad=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(panels) == 2:
        extra_handles, extra_labels = axes[0][1].get_legend_handles_labels()
        for handle, label in zip(extra_handles, extra_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
        handlelength=2.4,
    )
    fig.suptitle(concept, fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    curves = load()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    main_figure(curves, OUTPUT / "forgetting.png")
    grid_figure(
        curves,
        OUTPUT / "appendix-ranks.png",
        [
            ("baseline", GRAY, "baseline"),
            ("rank 1", TEAL, "rank 1"),
            ("rank 2", SAND, "rank 2"),
            ("full", ORANGE, "full"),
        ],
    )
    grid_figure(
        curves,
        OUTPUT / "appendix-frozen.png",
        [
            ("baseline", GRAY, "baseline"),
            ("rank 1, frozen", TEAL, "rank 1, frozen"),
            ("rank 2, frozen", SAND, "rank 2, frozen"),
            ("full, frozen", ORANGE, "full, frozen"),
        ],
    )
    chinese_figure(curves, OUTPUT / "appendix-chinese.png")
    extras = [
        "fairytale",
        "numbered",
        "theistic",
        "probabilistic",
        "fairytale, residual scale 0.5",
        "theistic, residual scale 1",
    ]
    for concept in extras:
        slug = concept.replace(", ", "-").replace(" ", "-")
        concept_figure(curves, concept, OUTPUT / f"appendix-{slug}.png")
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
