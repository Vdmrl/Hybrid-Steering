"""Concept-versus-benchmark plots in the same layout as the GDN tradeoff pages."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {
    "baseline": "#475569",
    "add": "#2563eb",
    "clamp": "#15803d",
    "rank1": "#d62728",
    "rank2": "#9467bd",
    "full": "#15803d",
}
WIDTH = 760
HEIGHT = 480
PLOT_LEFT = 88.7
PLOT_RIGHT = 557.3
AXIS_LEFT = 70
AXIS_RIGHT = 576
PLOT_TOP = 32
PLOT_BOTTOM = 434


def radius(scale: float) -> float:
    """Marker area grows with scale. Scale 0 uses the step below 0.25."""
    return 2.4 + 1.6 * max(scale, 0.125)


def _panels(pipeline: Path, kind: str, prefix: str) -> list[dict]:
    panels = []
    for run_dir in sorted(pipeline.glob(f"{prefix}-*-{kind}")):
        language = run_dir.name.removeprefix(f"{prefix}-").removesuffix(f"-{kind}")
        for report_path in sorted(run_dir.glob("reports/*/report.json")):
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            rows = payload["rows"]
            if not rows:
                continue
            metric = rows[0]["benchmark_metric"]
            benchmark = "HumanEval" if metric == "pass_at_1" else "IFEval"
            panels.append(
                {
                    "language": language,
                    "benchmark": benchmark,
                    "metric": metric,
                    "rows": rows,
                }
            )
    return panels


def _y_label(metric: str) -> str:
    if metric == "pass_at_1":
        return "HumanEval, pass@1"
    if metric == "prompt_strict":
        return "IFEval, prompt-strict"
    return metric


def _nice_step(span: float) -> float:
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if span / step <= 6:
            return step
    return 1.0


def _svg(rows: list[dict], title: str, condition: str) -> str:
    selected = [row for row in rows if row["condition"] in {condition, "baseline"}]
    lows = [row["benchmark_low"] for row in selected]
    highs = [row["benchmark_high"] for row in selected]
    step = _nice_step(max(highs) - min(lows) + 1e-6)
    y_min = 0.0
    y_max = max(step, (int(max(highs) / step) + 1) * step)
    color = COLORS.get(condition, "#15803d")

    def x_of(value: float) -> float:
        return PLOT_LEFT + (PLOT_RIGHT - PLOT_LEFT) * value

    def y_of(value: float) -> float:
        return PLOT_BOTTOM - (PLOT_BOTTOM - PLOT_TOP) * (value - y_min) / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
        f"<title>{escape(title)}</title>",
        "<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#1f2937}"
        ".grid{stroke:#e7edf4;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.25}"
        ".tick{font-size:12px;fill:#475569}.axis-title{font-size:13px;fill:#1f2937}"
        ".panel-title{font-size:16px;font-weight:650;fill:#111827}"
        ".scale-label{font-size:12px;font-weight:650}.legend{font-size:13px;fill:#1f2937}"
        ".legend-note{font-size:11px;fill:#64748b}</style>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text class="panel-title" x="70" y="22">{escape(title)}</text>',
    ]
    tick = y_min
    while tick <= y_max + 1e-9:
        y = y_of(tick)
        parts.append(
            f'<line class="grid" x1="{AXIS_LEFT}" y1="{y:.1f}" x2="{AXIS_RIGHT}" y2="{y:.1f}"/>'
        )
        label = f"{tick:.2f}".rstrip("0").rstrip(".")
        parts.append(f'<text class="tick" x="62" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
        tick += step
    for value in (0, 0.25, 0.5, 0.75, 1):
        x = x_of(value)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{PLOT_TOP}" x2="{x:.1f}" y2="{PLOT_BOTTOM}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="450" text-anchor="middle">{value:g}</text>'
        )
    parts += [
        f'<line class="axis" x1="{AXIS_LEFT}" y1="{PLOT_TOP}" x2="{AXIS_LEFT}" y2="{PLOT_BOTTOM}"/>',
        f'<line class="axis" x1="{AXIS_LEFT}" y1="{PLOT_BOTTOM}" x2="{AXIS_RIGHT}" y2="{PLOT_BOTTOM}"/>',
        '<text class="axis-title" x="323" y="470" text-anchor="middle">Concept expression</text>',
        f'<text class="axis-title" x="16" y="233" text-anchor="middle" transform="rotate(-90 16 233)">{escape(_y_label(selected[0]["benchmark_metric"]))}</text>',
    ]
    steered = sorted(
        (row for row in selected if row["condition"] == condition), key=lambda row: row["scale"]
    )
    if len(steered) > 1:
        points = " ".join(
            f"{x_of(row['judge_mean']):.1f},{y_of(row['benchmark_score']):.1f}" for row in steered
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.75" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def cross(row: dict, stroke: str) -> None:
        x, y = x_of(row["judge_mean"]), y_of(row["benchmark_score"])
        x0, x1 = x_of(row["judge_low"]), x_of(row["judge_high"])
        y0, y1 = y_of(row["benchmark_low"]), y_of(row["benchmark_high"])
        parts.extend(
            [
                f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1.25"/>',
                f'<line x1="{x0:.1f}" y1="{y - 3:.1f}" x2="{x0:.1f}" y2="{y + 3:.1f}" stroke="{stroke}" stroke-width="1.25"/>',
                f'<line x1="{x1:.1f}" y1="{y - 3:.1f}" x2="{x1:.1f}" y2="{y + 3:.1f}" stroke="{stroke}" stroke-width="1.25"/>',
                f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="{stroke}" stroke-width="1.25"/>',
                f'<line x1="{x - 3:.1f}" y1="{y0:.1f}" x2="{x + 3:.1f}" y2="{y0:.1f}" stroke="{stroke}" stroke-width="1.25"/>',
                f'<line x1="{x - 3:.1f}" y1="{y1:.1f}" x2="{x + 3:.1f}" y2="{y1:.1f}" stroke="{stroke}" stroke-width="1.25"/>',
            ]
        )

    for row in selected:
        stroke = COLORS["baseline"] if row["condition"] == "baseline" else color
        cross(row, stroke)
    for row in sorted(selected, key=lambda item: item["scale"]):
        x, y = x_of(row["judge_mean"]), y_of(row["benchmark_score"])
        stroke = COLORS["baseline"] if row["condition"] == "baseline" else color
        tip = (
            f"{row['condition']}, scale {row['scale']:g}: concept {row['judge_mean']:.3f} "
            f"[{row['judge_low']:.3f}, {row['judge_high']:.3f}], "
            f"{_y_label(row['benchmark_metric'])} {row['benchmark_score']:.3f} "
            f"[{row['benchmark_low']:.3f}, {row['benchmark_high']:.3f}]"
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius(row["scale"]):.1f}" fill="{stroke}">'
            f"<title>{escape(tip)}</title></circle>"
        )
        anchor = "end" if row["judge_mean"] > 0.75 else "start"
        shift = -8 if anchor == "end" else 8
        parts.append(
            f'<text class="scale-label" x="{x + shift:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" '
            f'fill="{stroke}">{row["scale"]:g}</text>'
        )
    parts += [
        '<rect x="592" y="36" width="156" height="117" rx="4" fill="#ffffff" stroke="#d9e2ec"/>',
        '<circle cx="619" cy="54" r="3.6" fill="#475569"/>',
        '<text class="legend" x="644" y="58">baseline</text>',
        f'<line x1="602" y1="78" x2="636" y2="78" stroke="{color}" stroke-width="2"/>',
        f'<circle cx="619" cy="78" r="3.6" fill="{color}"/>',
        f'<text class="legend" x="644" y="82">{escape(condition)}</text>',
        '<line x1="602" y1="96" x2="738" y2="96" stroke="#e5e7eb"/>',
        '<text class="legend-note" x="602" y="112">number = scale</text>',
        '<text class="legend-note" x="602" y="127">size ∝ scale</text>',
        '<text class="legend-note" x="602" y="142">cross = 95% CI</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def _save_tradeoff(rows: list[dict], title: str, condition: str, stem: Path) -> None:
    plt.switch_backend("Agg")
    selected = [row for row in rows if row["condition"] in {condition, "baseline"}]
    lows = [row["benchmark_low"] for row in selected]
    highs = [row["benchmark_high"] for row in selected]
    step = _nice_step(max(highs) - min(lows) + 1e-6)
    y_max = max(step, (int(max(highs) / step) + 1) * step)
    color = COLORS.get(condition, "#15803d")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    steered = sorted(
        (row for row in selected if row["condition"] == condition), key=lambda row: row["scale"]
    )
    if len(steered) > 1:
        ax.plot(
            [row["judge_mean"] for row in steered],
            [row["benchmark_score"] for row in steered],
            color=color,
            linewidth=1.75,
        )
    for row in selected:
        stroke = COLORS["baseline"] if row["condition"] == "baseline" else color
        ax.errorbar(
            row["judge_mean"],
            row["benchmark_score"],
            xerr=[
                [row["judge_mean"] - row["judge_low"]],
                [row["judge_high"] - row["judge_mean"]],
            ],
            yerr=[
                [max(0.0, row["benchmark_score"] - row["benchmark_low"])],
                [max(0.0, row["benchmark_high"] - row["benchmark_score"])],
            ],
            fmt="o",
            color=stroke,
            markersize=radius(row["scale"]),
            capsize=3,
        )
        ax.annotate(
            f"{row['scale']:g}",
            (row["judge_mean"], row["benchmark_score"]),
            textcoords="offset points",
            xytext=(-8, 0) if row["judge_mean"] > 0.75 else (8, 0),
            ha="right" if row["judge_mean"] > 0.75 else "left",
            va="center",
            color=stroke,
            fontsize=9,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_max)
    ax.set_xticks((0, 0.25, 0.5, 0.75, 1))
    ax.set_xlabel("Concept expression")
    ax.set_ylabel(_y_label(selected[0]["benchmark_metric"]))
    ax.set_title(title)
    ax.grid(True, color="#e7edf4")
    ax.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color=COLORS["baseline"], linestyle="", label="baseline"
            ),
            plt.Line2D([0], [0], marker="o", color=color, linestyle="-", label=condition),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=200)
    plt.close(fig)


def write_svgs(pipeline: Path, kind: str, output: Path, prefix: str = "qwen") -> list[Path]:
    """Write one SVG per language, condition, and benchmark."""
    panels = _panels(pipeline, kind, prefix)
    if kind == "residual":
        conditions = ("add", "clamp")
    elif kind == "mamba-add":
        conditions = ("rank1", "rank2", "full")
    else:
        conditions = ("clamp",)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for benchmark in ("IFEval", "HumanEval"):
        languages = []
        for panel in panels:
            if panel["benchmark"] == benchmark and panel["language"] not in languages:
                languages.append(panel["language"])
        for language in languages:
            rows = next(
                panel["rows"]
                for panel in panels
                if panel["benchmark"] == benchmark and panel["language"] == language
            )
            present = {row["condition"] for row in rows}
            for condition in conditions:
                if condition not in present:
                    continue
                title = f"{language.capitalize()}, {condition}, {benchmark}"
                path = output / f"{prefix}-{kind}-{language}-{condition}-{benchmark.lower()}"
                _save_tradeoff(rows, title, condition, path)
                written.append(Path(f"{path}.svg"))
    return written


def write_tradeoff(pipeline: Path, kind: str, output: Path, prefix: str = "qwen") -> Path:
    """Write one HTML page. ``kind`` selects the run directories and their conditions."""
    panels = _panels(pipeline, kind, prefix)
    if not panels:
        raise FileNotFoundError(f"no {prefix} {kind} reports under {pipeline}")
    model = "Falcon-H1-7B" if prefix == "falcon" else "Qwen3.5-9B"
    heading = "residual" if kind == "residual" else kind.replace("-", " ")
    if kind == "residual":
        conditions = ("add", "clamp")
    elif kind == "mamba-add":
        conditions = ("rank1", "rank2", "full")
    else:
        conditions = ("clamp",)
    sections = []
    for benchmark in ("IFEval", "HumanEval"):
        block = [f"<h2>{benchmark}</h2>"]
        languages = []
        for panel in panels:
            if panel["benchmark"] == benchmark and panel["language"] not in languages:
                languages.append(panel["language"])
        for language in languages:
            rows = next(
                panel["rows"]
                for panel in panels
                if panel["benchmark"] == benchmark and panel["language"] == language
            )
            present = {row["condition"] for row in rows}
            figures = []
            for condition in conditions:
                if condition not in present:
                    continue
                title = f"{language.capitalize()}, {condition}"
                figures.append(f"<figure>{_svg(rows, title, condition)}</figure>")
            if figures:
                block.append(
                    f"<section><h3>{language.capitalize()}</h3>"
                    f'<div class="grid">{"".join(figures)}</div></section>'
                )
        sections.append("".join(block))
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{model} {heading}: concept vs benchmark</title>
<style>
  body {{ margin: 0; background: #f4f6f8; color: #1f2937; font: 15px/1.45 ui-sans-serif, system-ui, sans-serif; }}
  main {{ max-width: 1800px; margin: 0 auto; padding: 28px 24px 48px; }}
  h1 {{ margin: 0 0 8px; font-size: 22px; font-weight: 650; }}
  h2 {{ margin: 28px 0 8px; font-size: 18px; }}
  h3 {{ margin: 14px 0 8px; font-size: 15px; font-weight: 650; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
  figure {{ margin: 0; background: #fff; border: 1px solid #e3e8ef; }}
  svg {{ display: block; width: 100%; height: auto; }}
</style>
</head>
<body>
<main>
  <h1>{model}, {heading}</h1>
  {"".join(sections)}
</main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output
