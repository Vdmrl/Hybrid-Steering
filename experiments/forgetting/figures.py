"""One rank-1 decay figure per concept.

The left axis is the mean concept score: Lingua 0/1, judge 0-4. Steered and
unsteered answers are separate lines, so the effect is the gap between them.
The axis tops out just above that concept's highest score. Concepts are not
averaged or overlaid. The right axis is the GDN state ratio. Intervals are a
95% bootstrap over questions.

``r(L) = ||dS(L)|| / ||dS(0)||`` is the state difference at the last prompt
token, from ``heads.pt``, averaged over fillers that have both lengths.

``short`` and ``long`` next to ``normal`` add filler lengths to that curve.
``frozen`` is the run that reuses the unsteered attention write.

    uv run python experiments/forgetting/figures.py --output runs/paper-figures
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from hybrid_steering.runtime import read_jsonl

PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"
METHOD = "rank1"
MIN_GAIN = 0.2
ROUNDS = 1000
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
LANGUAGES = {"en-ru", "en-fr", "en-zh", "en-ar"}
NORMAL = "#1f77b4"
CLEAN = "#d62728"
FROZEN = "#9467bd"
BASELINE = "#222222"
STATE = "#888888"
EXTRA = ("short", "long")


class Run:
    """One filler-decay curve: raw per-question scores and state ratios."""

    def __init__(self, paths: list[Path]) -> None:
        self.meta = json.loads((paths[0] / "summary.json").read_text())
        rows = [row for path in paths for row in read_jsonl(path / "rows.jsonl")]
        self.lengths = sorted({r["length"] for r in rows})
        self.questions = sorted({r["index"] for r in rows})
        self.scale = self.meta["grid"][METHOD][0]
        grouped: dict = {}
        for r in rows:
            if r["method"] not in ("baseline", METHOD):
                continue
            if r["method"] == METHOD and r["scale"] != self.scale:
                continue
            if r.get("evaluable") is False:
                value = 0.0
            elif r.get("concept_score") is None:
                continue
            else:
                value = r["concept_score"]
            grouped.setdefault((r["method"], r["length"]), {}).setdefault(r["index"], []).append(
                value
            )
        self.scores = {
            key: {i: sum(v) / len(v) for i, v in by_question.items()}
            for key, by_question in grouped.items()
        }
        cells: dict = {}
        for path in paths:
            for key, value in torch.load(path / "heads.pt")["cells"].items():
                cells.setdefault(key, value)
        by_filler: dict[int, dict[int, float]] = {}
        for (method, scale, filler, length), cell in cells.items():
            if method == METHOD and scale == self.scale:
                by_filler.setdefault(filler, {})[length] = float(cell["state_delta"].norm())
        self.ratio = {}
        for length in self.lengths:
            vals = [
                norms[length] / norms[0]
                for norms in by_filler.values()
                if length in norms and norms.get(0)
            ]
            self.ratio[length] = sum(vals) / len(vals) if vals else math.nan

    def mean(self, method: str, length: int, sample: list[int]) -> float:
        table = self.scores.get((method, length), {})
        values = [table[i] for i in sample if i in table]
        return sum(values) / len(values) if values else math.nan

    def gain(self, length: int, sample: list[int]) -> float:
        return self.mean(METHOD, length, sample) - self.mean("baseline", length, sample)


def interval(values: list[float]) -> tuple[float, float]:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return math.nan, math.nan
    return finite[int(0.025 * len(finite))], finite[min(len(finite) - 1, int(0.975 * len(finite)))]


def estimate(function, full: list[int], draws: list[list[int]]) -> tuple[float, float, float]:
    low, high = interval([function(sample) for sample in draws])
    return function(full), low, high


def band(x, estimates, color, name, dash="solid", width=4, group=None) -> list[dict]:
    mean, low, high = (list(column) for column in zip(*estimates, strict=True))
    common = {"legendgroup": group or name, "showlegend": False, "hoverinfo": "skip"}
    return [
        {"x": x, "y": high, "mode": "lines", "line": {"width": 0}, **common},
        {
            "x": x,
            "y": low,
            "mode": "lines",
            "line": {"width": 0},
            "fill": "tonexty",
            "fillcolor": "rgba({},{},{},0.18)".format(
                *(int(color[i : i + 2], 16) for i in (1, 3, 5))
            ),
            **common,
        },
        {
            "x": x,
            "y": mean,
            "mode": "lines+markers",
            "name": name,
            "legendgroup": group or name,
            "line": {"color": color, "width": width, "dash": dash},
            "marker": {"size": 9},
        },
    ]


def figure(name: str, title: str, caption: str, traces: list[dict], layout: dict) -> dict:
    base = {
        "width": 900,
        "height": 560,
        "margin": {"l": 80, "r": 30, "t": 30, "b": 70},
        "font": {"size": 16, "family": "Arial"},
        "plot_bgcolor": "white",
        "legend": {"font": {"size": 14}},
    }
    layout = {**base, **layout}
    for axis in ("xaxis", "yaxis", "yaxis2"):
        if axis not in layout and axis == "yaxis2":
            continue
        layout[axis] = {
            "gridcolor": "#e5e5e5",
            "linecolor": "#444",
            "showline": True,
            "mirror": axis != "yaxis" or "yaxis2" not in layout,
            "zeroline": False,
            "showgrid": axis != "yaxis2",
            **layout.get(axis, {}),
        }
    payload = json.dumps(traces).replace("NaN", "null").replace("Infinity", "null")
    config = json.dumps(
        {"toImageButtonOptions": {"format": "svg", "filename": name}, "displaylogo": False}
    )
    script = f"Plotly.newPlot('{name}',{payload},{json.dumps(layout)},{config});"
    return {"name": name, "title": title, "caption": caption, "script": script}


def page(title: str, body: str) -> str:
    return (
        f"<!doctype html><meta charset='utf-8'><title>{title}</title>"
        f"<script src='{PLOTLY}'></script><style>body{{font:15px/1.5 system-ui,sans-serif;"
        "max-width:960px;margin:1.5em auto;padding:0 1em;color:#222}table{border-collapse:"
        "collapse}td,th{border:1px solid #ccc;padding:4px 9px}th{background:#f3f3f3}"
        "h2{margin-top:2.2em}</style>" + body
    )


def section(item: dict, link: bool) -> str:
    title = f"<a href='{item['name']}.html'>{item['title']}</a>" if link else item["title"]
    return (
        f"<h2>{title}</h2><p>{item['caption']}</p><div id='{item['name']}'></div>"
        f"<script>{item['script']}</script>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/forgetting"))
    parser.add_argument("--output", type=Path, default=Path("runs/forgetting/figures"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def present(slug: str, name: str) -> list[Path]:
        path = args.runs / slug / name
        return [path] if (path / "summary.json").exists() else []

    runs = {
        slug: {
            "normal": Run(
                present(slug, "normal") + [p for name in EXTRA for p in present(slug, name)]
            ),
            "clean": Run(present(slug, "clean")),
            **({"frozen": Run(present(slug, "frozen"))} if present(slug, "frozen") else {}),
        }
        for slug in LABELS
    }
    full = runs["en-ru"]["normal"].questions
    rng = random.Random(args.seed)
    draws = [rng.choices(full, k=len(full)) for _ in range(ROUNDS)]
    shown = [
        s
        for s in LABELS
        if runs[s]["normal"].gain(0, full) / (1 if s in LANGUAGES else 4) > MIN_GAIN
    ]
    dropped = [LABELS[s] for s in LABELS if s not in shown]

    def score_band(run: Run, method: str, lengths: list[int]) -> list[tuple[float, float, float]]:
        have = set(run.lengths)
        return [
            estimate(lambda sample, L=L: run.mean(method, L, sample), full, draws)
            if L in have
            else (math.nan, math.nan, math.nan)
            for L in lengths
        ]

    figures = []
    for slug in shown:
        normal = runs[slug]["normal"]
        clean = runs[slug]["clean"]
        frozen = runs[slug].get("frozen")
        lengths = sorted(
            set(normal.lengths) | set(clean.lengths) | set(frozen.lengths if frozen else ())
        )
        x = [L + 1 for L in lengths]
        log_x = {
            "type": "log",
            "title": "filler tokens between question and answer",
            "tickvals": x,
            "ticktext": [str(L) for L in lengths],
        }
        steered = score_band(normal, METHOD, lengths)
        steered_clean = score_band(clean, METHOD, lengths)
        baseline = score_band(normal, "baseline", lengths)
        series = [steered, steered_clean, baseline]
        if frozen:
            series.append(score_band(frozen, METHOD, lengths))
        peak = max(point[0] for points in series for point in points if math.isfinite(point[0]))
        ceiling = 1 if slug in LANGUAGES else 4
        top = (
            ceiling * 1.05 if peak >= 0.9 * ceiling else max(0.25, math.ceil(peak * 1.25 * 20) / 20)
        )
        unit = "0/1" if slug in LANGUAGES else "0–4"
        traces = band(x, steered, NORMAL, "steered")
        traces += band(x, steered_clean, CLEAN, "steered, clean KV")
        if frozen:
            traces += band(x, series[-1], FROZEN, "steered, frozen attn")
        traces += band(x, baseline, BASELINE, "unsteered", width=2)
        traces.append(
            {
                "x": x,
                "y": [normal.ratio.get(L, math.nan) for L in lengths],
                "yaxis": "y2",
                "mode": "lines+markers",
                "name": "GDN state left",
                "line": {"color": STATE, "width": 2, "dash": "dash"},
                "marker": {"size": 8, "color": STATE},
            }
        )
        figures.append(
            figure(
                slug,
                f"{LABELS[slug]}, rank 1, scale {normal.scale:g}",
                f"Left axis: mean concept score ({unit}). The effect is the gap between the "
                "steered line and the unsteered baseline. The axis ends just above this "
                "concept's highest score. Right axis: GDN state difference at the last prompt "
                "token, relative to L = 0. Clean KV reuses unsteered keys and values. "
                + (
                    "Frozen attn reuses unsteered q, k, and v, so the attention write matches "
                    "the baseline. "
                    if frozen
                    else ""
                )
                + "Bands: 95% bootstrap.",
                traces,
                {
                    "xaxis": log_x,
                    "yaxis": {"title": "concept score", "range": [0, top]},
                    "yaxis2": {
                        "title": "‖ΔS(L)‖ / ‖ΔS(0)‖",
                        "overlaying": "y",
                        "side": "right",
                        "range": [0, 1.05],
                    },
                    "title": {"text": LABELS[slug], "font": {"size": 22}},
                    "legend": {
                        "orientation": "h",
                        "x": 0.5,
                        "xanchor": "center",
                        "y": -0.22,
                        "yanchor": "top",
                    },
                    "margin": {"l": 80, "r": 90, "t": 60, "b": 120},
                },
            )
        )

    lengths = sorted({L for run in runs.values() for L in run["normal"].lengths})

    def gain_cell(run: Run | None, length: int) -> str:
        if run is None or length not in run.lengths:
            return "–"
        return f"{run.gain(length, full):.2f}"

    body = ""
    for s in LABELS:
        run = runs[s]["normal"]
        clean = runs[s].get("clean")
        body += (
            f"<tr><td>{LABELS[s]}</td><td>{run.scale:g}</td>"
            + "".join(f"<td>{gain_cell(run, L)} / {gain_cell(clean, L)}</td>" for L in lengths)
            + "</tr>"
        )
    table = (
        "<h2>Table 1. Rank-1 scale and concept gain, normal / clean KV</h2><p>Scale chosen at "
        "L = 0 on the tune split: highest concept rate with quality ≥ baseline − 0.5 (− 0.7 for "
        "languages) and no extra repetition. Gains are mean score differences on the held "
        "split: Lingua 0/1, judge 0–4. 50 questions × 2 fillers.</p><table><tr><th>concept</th>"
        "<th>scale</th>" + "".join(f"<th>L = {L}</th>" for L in lengths) + f"</tr>{body}</table>"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("fig*.html"):
        stale.unlink()
    for item in figures:
        (args.output / f"{item['name']}.html").write_text(
            page(item["title"], section(item, link=False)), encoding="utf-8"
        )
    (args.output / "index.html").write_text(
        page(
            "Paper figures: rank-1 decay",
            "<h1>Rank-1 steering decay over filler tokens</h1><p>One concept per plot. "
            "Left axis is the score itself, with the unsteered baseline on the same axis; "
            "the top of the axis follows that concept. "
            f"Not shown (gain at L = 0 below {MIN_GAIN}): {', '.join(dropped)}. "
            "Camera icon saves SVG; titles open the plot as its own page.</p>"
            + "".join(section(item, link=True) for item in figures)
            + table,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
