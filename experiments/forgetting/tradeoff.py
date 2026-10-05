"""Two views of the L = 0 tune grid: paths, then the chosen points.

Rank 1, rank 2 and full are drawn as paths so the three curves can be compared
without a label on every scale. A second plot keeps only the scale chosen for
the decay runs, with a 95% bootstrap cross on both axes. Clamp and release are
omitted: they are a different intervention.

    uv run python experiments/forgetting/tradeoff.py --output runs/forgetting/scale/tradeoff.html
"""

from __future__ import annotations

import argparse
import json
import zlib
from html import escape
from pathlib import Path

from hybrid_steering.runtime import import_path, read_jsonl

report = import_path(Path(__file__).with_name("report.py"))
COLORS, PLOTLY, bootstrap = report.COLORS, report.PLOTLY, report.bootstrap

LABELS = {
    "en-ru": "Russian",
    "en-fr": "French",
    "en-zh": "Chinese",
    "en-ar": "Arabic",
    "plain-technical_language": "technical",
    "plain-theistic_framing": "theistic",
    "plain-comparative_framing": "comparative",
    "plain-probabilistic_framing": "probabilistic",
    "plain-fictional_narrative": "fictional",
}


def cells(root: Path, slug: str) -> list[dict]:
    rows = []
    for path in sorted((root / slug).rglob("rows.jsonl")):
        for row in read_jsonl(path):
            if row.get("length", 0) != 0 or row.get("filler", 0) != 0:
                continue
            if row.get("hit") is None or row.get("content_quality") is None:
                continue
            rows.append(row)
    chosen = {}
    chosen_path = root / slug / "chosen.json"
    if chosen_path.exists():
        chosen = json.loads(chosen_path.read_text())["chosen"]
    grouped: dict[tuple, list] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["scale"]), []).append(row)
    points = []
    for (method, scale), group in sorted(grouped.items()):
        seed = zlib.crc32(f"{slug}:{method}:{scale}".encode())
        rate, rate_low, rate_high = bootstrap(group, "hit", seed=seed)
        quality, quality_low, quality_high = bootstrap(group, "content_quality", seed=seed + 1)
        points.append(
            {
                "method": method,
                "scale": scale,
                "rate": rate,
                "rate_low": rate_low,
                "rate_high": rate_high,
                "quality": quality,
                "quality_low": quality_low,
                "quality_high": quality_high,
                "chosen": chosen.get(method, {}).get("scale") == scale,
            }
        )
    return points


ADD = ("rank1", "rank2", "full")
NAMES = {"rank1": "rank 1", "rank2": "rank 2", "full": "full"}
# Pixel shift so concept names do not sit on the chosen points.
SHIFT = {
    "Russian": (-8, 26),
    "French": (-78, 8),
    "Arabic": (62, 22),
    "Chinese": (36, -8),
    "technical": (28, -30),
    "theistic": (-72, 0),
    "comparative": (48, 12),
    "probabilistic": (-62, -10),
    "fictional": (52, -16),
}
DASH = {"rank1": "solid", "rank2": "dash", "full": "dot"}


def _ordered(points: list[dict], method: str) -> list[dict]:
    return sorted((p for p in points if p["method"] == method), key=lambda p: p["scale"])


def _bars(series: list[dict], arm: str) -> dict:
    high, low, center = f"{arm}_high", f"{arm}_low", "rate" if arm == "rate" else "quality"
    return {
        "type": "data",
        "symmetric": False,
        "array": [p[high] - p[center] for p in series],
        "arrayminus": [p[center] - p[low] for p in series],
        "thickness": 1.4,
        "width": 0,
        "color": "#444",
    }


def path_traces(points: list[dict]) -> tuple[list[dict], dict | None]:
    """Lines through increasing scale. The ring is the chosen scale. No error bars."""
    drawn = []
    for method in ADD:
        series = _ordered(points, method)
        if not series:
            continue
        drawn.append(
            {
                "x": [p["quality"] for p in series],
                "y": [p["rate"] for p in series],
                "customdata": [p["scale"] for p in series],
                "mode": "lines+markers",
                "name": NAMES[method],
                "line": {"color": COLORS[method], "width": 2.5, "dash": DASH[method]},
                "marker": {"color": COLORS[method], "size": 5},
                "hovertemplate": (
                    NAMES[method] + " ×%{customdata:g}<br>quality %{x:.2f}"
                    "<br>concept rate %{y:.2f}<extra></extra>"
                ),
            }
        )
        chosen = [p for p in series if p["chosen"]]
        if chosen:
            drawn.append(
                {
                    "x": [p["quality"] for p in chosen],
                    "y": [p["rate"] for p in chosen],
                    "customdata": [p["scale"] for p in chosen],
                    "mode": "markers",
                    "showlegend": False,
                    "hoverinfo": "skip",
                    "marker": {
                        "color": COLORS[method],
                        "size": 13,
                        "symbol": "circle",
                        "line": {"color": "#111", "width": 1.5},
                    },
                }
            )
    base = _ordered(points, "baseline")
    baseline = base[0] if base else None
    if baseline:
        drawn.append(
            {
                "x": [baseline["quality"]],
                "y": [baseline["rate"]],
                "mode": "markers",
                "name": "baseline",
                "marker": {"color": "#111", "size": 12, "symbol": "x"},
                "hovertemplate": "baseline<br>quality %{x:.2f}<br>concept rate %{y:.2f}<extra></extra>",
            }
        )
    return drawn, baseline


def optima_traces(named: list[tuple[str, list[dict]]]) -> tuple[list[dict], list[dict]]:
    """One cross per chosen scale, plus a name at the middle of the three methods."""
    drawn = []
    notes = []
    bases = []
    for method in ADD:
        series = []
        for _label, points in named:
            series.extend(p for p in _ordered(points, method) if p["chosen"])
        if not series:
            continue
        drawn.append(
            {
                "x": [p["quality"] for p in series],
                "y": [p["rate"] for p in series],
                "customdata": [p["scale"] for p in series],
                "error_x": _bars(series, "quality"),
                "error_y": _bars(series, "rate"),
                "mode": "markers",
                "name": NAMES[method],
                "marker": {
                    "color": COLORS[method],
                    "size": 11,
                    "line": {"color": "#fff", "width": 1},
                },
                "hovertemplate": (
                    NAMES[method] + " ×%{customdata:g}<br>quality %{x:.2f}"
                    "<br>concept rate %{y:.2f}<extra></extra>"
                ),
            }
        )
    for label, points in named:
        chosen = [p for method in ADD for p in _ordered(points, method) if p["chosen"]]
        base = _ordered(points, "baseline")
        if base:
            bases.append(base[0])
        if not chosen:
            continue
        cx = sum(p["quality"] for p in chosen) / len(chosen)
        cy = sum(p["rate"] for p in chosen) / len(chosen)
        dx, dy = SHIFT.get(label, (0, 12))
        notes.append(
            {
                "x": cx,
                "y": cy,
                "text": label,
                "showarrow": True,
                "ax": dx,
                "ay": -dy,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": "#b5b5b5",
                "font": {"size": 13, "color": "#1c1c1a"},
                "bgcolor": "rgba(255,255,255,0.9)",
            }
        )
    if bases:
        drawn.insert(
            0,
            {
                "x": [p["quality"] for p in bases],
                "y": [p["rate"] for p in bases],
                "mode": "markers",
                "name": "baseline",
                "marker": {"color": "#111", "size": 11, "symbol": "x"},
                "hovertemplate": "baseline<br>quality %{x:.2f}<br>concept rate %{y:.2f}<extra></extra>",
            },
        )
    return drawn, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/forgetting/scale"))
    parser.add_argument("--output", type=Path, default=Path("runs/forgetting/scale/tradeoff.html"))
    args = parser.parse_args()
    slugs = [slug for slug in LABELS if (args.runs / slug).is_dir()]
    named = [(LABELS[slug], cells(args.runs, slug)) for slug in slugs]
    path_range = {
        "xaxis": {"title": "content quality", "range": [0, 2.7], "dtick": 0.5},
        "yaxis": {"title": "concept rate", "range": [-0.03, 1.08], "dtick": 0.25},
    }
    sections = []
    for index, (label, points) in enumerate(named):
        plot, baseline = path_traces(points)
        shapes = []
        if baseline:
            shapes = [
                {
                    "type": "line",
                    "x0": baseline["quality"],
                    "x1": baseline["quality"],
                    "y0": 0,
                    "y1": 1,
                    "line": {"color": "#d5d5d5", "width": 1, "dash": "dot"},
                },
                {
                    "type": "line",
                    "x0": 0,
                    "x1": 2.7,
                    "y0": baseline["rate"],
                    "y1": baseline["rate"],
                    "line": {"color": "#d5d5d5", "width": 1, "dash": "dot"},
                },
            ]
        layout = {
            **path_range,
            "shapes": shapes,
            "title": {"text": label, "font": {"size": 15}, "x": 0.02, "xanchor": "left"},
            "margin": {"t": 32, "r": 12, "b": 40, "l": 44},
            "showlegend": False,
            "height": 300,
        }
        sections.append(
            f"<div id='p{index}'></div><script>Plotly.newPlot('p{index}',"
            f"{json.dumps(plot)},{json.dumps(layout)});</script>"
        )
    optima, notes = optima_traces(named)
    optima_layout = {
        "xaxis": {"title": "content quality", "range": [1.15, 2.55], "dtick": 0.2},
        "yaxis": {"title": "concept rate", "range": [-0.06, 1.14], "dtick": 0.2},
        "annotations": notes,
        "legend": {"orientation": "h", "y": 1.08},
        "margin": {"t": 36, "r": 16, "b": 48, "l": 52},
        "height": 640,
    }
    legend = "".join(
        f"<li><i style='background:{COLORS[method]}'></i>{escape(NAMES[method])}"
        f" ({DASH[method]})</li>"
        for method in ADD
    )
    page = (
        "<!doctype html><meta charset='utf-8'><title>L = 0 tradeoff</title>"
        f"<script src='{PLOTLY}'></script><style>"
        "body{font:15px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1100px;color:#1c1c1a}"
        ".legend{display:flex;gap:16px;list-style:none;padding:0;margin:8px 0 4px}"
        ".legend i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px}"
        ".grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px 8px}"
        "@media(max-width:900px){.grid{grid-template-columns:1fr}}"
        "h1{font-size:22px;margin-bottom:4px}h2{font-size:18px;margin:28px 0 4px}"
        "p{max-width:760px}</style>"
        "<h1>Concept rate against quality at L = 0</h1>"
        "<p>Tune split, 50 questions, no filler. Only the one-shot add methods. "
        "Up and to the right is more of the concept at higher quality. "
        "The black cross is the unsteered baseline; dotted lines run through it.</p>"
        f"<ul class='legend'>{legend}<li><i style='background:#111;border-radius:0'></i>baseline</li></ul>"
        "<h2>The three ranks follow one path</h2>"
        "<p>Each line walks from the smallest scale, beside the baseline, to the largest. "
        "Rank 1 is solid, rank 2 is dashed, full is dotted, so a shared path stays visible. "
        "The ring is the scale kept for the decay runs. Hover a point for its scale.</p>"
        "<div class='grid'>" + "".join(sections) + "</div>"
        "<h2>The kept scale sits in a different place for each concept</h2>"
        "<p>Only the ring from the panels above. The cross is a 95% bootstrap over questions. "
        "This axis is zoomed to where those points actually are.</p>"
        f"<div id='optima'></div><script>Plotly.newPlot('optima',{json.dumps(optima)},"
        f"{json.dumps(optima_layout)});</script>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
