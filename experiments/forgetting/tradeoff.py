"""Concept rate against quality at L = 0, one cross per method and scale.

Both arms are a 95% bootstrap over questions. The tune grids in
``runs/forgetting/scale`` supply the points, including the release fine grid.

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


def traces(points: list[dict]) -> list[dict]:
    drawn = []
    for method in ["baseline", "rank1", "rank2", "full", "clamp", "release"]:
        series = sorted((p for p in points if p["method"] == method), key=lambda p: p["scale"])
        if not series:
            continue
        drawn.append(
            {
                "x": [p["quality"] for p in series],
                "y": [p["rate"] for p in series],
                "error_x": {
                    "type": "data",
                    "symmetric": False,
                    "array": [p["quality_high"] - p["quality"] for p in series],
                    "arrayminus": [p["quality"] - p["quality_low"] for p in series],
                    "thickness": 1,
                    "width": 3,
                },
                "error_y": {
                    "type": "data",
                    "symmetric": False,
                    "array": [p["rate_high"] - p["rate"] for p in series],
                    "arrayminus": [p["rate"] - p["rate_low"] for p in series],
                    "thickness": 1,
                    "width": 3,
                },
                "mode": "lines+markers+text",
                "text": [f"{p['scale']:g}" + (" *" if p["chosen"] else "") for p in series],
                "textposition": "top center",
                "textfont": {"size": 10},
                "name": method,
                "legendgroup": method,
                "showlegend": True,
                "line": {"color": COLORS[method], "width": 1},
                "marker": {
                    "color": COLORS[method],
                    "size": [11 if p["chosen"] else 7 for p in series],
                    "symbol": ["diamond" if p["chosen"] else "circle" for p in series],
                },
                "hovertemplate": (
                    method + " ×%{text}<br>quality %{x:.2f}<br>concept rate %{y:.2f}<extra></extra>"
                ),
            }
        )
    return drawn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/forgetting/scale"))
    parser.add_argument("--output", type=Path, default=Path("runs/forgetting/scale/tradeoff.html"))
    args = parser.parse_args()
    slugs = [slug for slug in LABELS if (args.runs / slug).is_dir()]
    sections = []
    for index, slug in enumerate(slugs):
        plot = traces(cells(args.runs, slug))
        layout = {
            "xaxis": {"title": "content quality", "range": [0, 4]},
            "yaxis": {"title": "concept rate", "range": [-0.02, 1.05]},
            "margin": {"t": 24, "r": 16, "b": 48, "l": 48},
            "legend": {"orientation": "h", "y": 1.14},
            "height": 420,
        }
        sections.append(
            f"<section><h2>{escape(LABELS[slug])}</h2><div id='p{index}'></div>"
            f"<script>Plotly.newPlot('p{index}',{json.dumps(plot)},{json.dumps(layout)});</script>"
            "</section>"
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>L = 0 tradeoff</title>"
        f"<script src='{PLOTLY}'></script><style>"
        "body{font:15px/1.45 system-ui,sans-serif;margin:24px;color:#1c1c1a}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 24px}"
        "@media(max-width:1000px){.grid{grid-template-columns:1fr}}"
        "h1{font-size:20px}h2{font-size:16px;margin:0}</style>"
        "<h1>Concept rate vs quality at L = 0</h1>"
        "<p>Tune split, 50 questions, filler 0. Each point is one method and scale. "
        "The cross is a 95% bootstrap over questions on both axes. Labels are scales; "
        "a diamond and * mark the scale chosen for the decay runs. Release includes the "
        "fine grid where one was run. Up and to the right is a stronger effect at higher quality.</p>"
        "<div class='grid'>" + "".join(sections) + "</div>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
