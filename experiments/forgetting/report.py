"""Decay plots for one or more filler-decay runs of the same concept.

Behaviour: concept rate and content quality against filler length per method
and scale, with a 95% bootstrap interval over questions (fillers pooled).

Mechanics, per GDN head h: ``r_h(L) = ||dS_h(L)|| / ||dS_h(0)||`` at the last
prompt token. The decay time ``tau_h`` is the least-squares slope of
``-L / ln r_h(L)`` over L > 0, fitted on heads that carry at least 0.1% of the
L = 0 difference energy. Clamp is skipped: it does not decay.

    uv run python experiments/forgetting/report.py --runs runs/forgetting/en-ru/normal \\
        runs/forgetting/en-ru/clean --output runs/forgetting/en-ru/report.html
"""

from __future__ import annotations

import argparse
import json
import math
import random
from html import escape
from pathlib import Path

import torch

from hybrid_steering.runtime import import_path, read_jsonl

meta_kind = import_path(Path(__file__).with_name("meta.py")).meta_kind

PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"
MIN_SHARE = 1e-3
COLORS = {
    "baseline": "#7f7f7f",
    "rank1": "#d62728",
    "rank2": "#9467bd",
    "full": "#2ca02c",
    "clamp": "#ff7f0e",
    "release": "#17becf",
}


def bootstrap(rows: list[dict], field: str, rounds: int = 1000, seed: int = 0):
    """Mean of ``field`` with a 95% interval, resampling questions."""
    by_question: dict[int, list[float]] = {}
    for row in rows:
        if row.get(field) is not None:
            by_question.setdefault(row["index"], []).append(float(row[field]))
    if not by_question:
        return math.nan, math.nan, math.nan
    keys = list(by_question)

    def mean(sample):
        values = [v for key in sample for v in by_question[key]]
        return sum(values) / len(values)

    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(keys, k=len(keys))) for _ in range(rounds))
    return mean(keys), draws[int(0.025 * rounds)], draws[int(0.975 * rounds) - 1]


def behaviour_traces(runs: dict[str, list[dict]], field: str) -> list[dict]:
    traces = []
    for label, rows in runs.items():
        series = sorted({(r["method"], r["scale"]) for r in rows})
        for method, scale in series:
            points = []
            for length in sorted({r["length"] for r in rows}):
                cell = [
                    r
                    for r in rows
                    if r["method"] == method and r["scale"] == scale and r["length"] == length
                ]
                points.append((length, *bootstrap(cell, field)))
            name = "baseline" if method == "baseline" else f"{method} ×{scale:g}"
            traces.append(
                {
                    "x": [p[0] + 1 for p in points],
                    "y": [p[1] for p in points],
                    "error_y": {
                        "type": "data",
                        "symmetric": False,
                        "array": [p[3] - p[1] for p in points],
                        "arrayminus": [p[1] - p[2] for p in points],
                    },
                    "name": f"{label}: {name}",
                    "legendgroup": name,
                    "mode": "lines+markers",
                    "line": {
                        "dash": "dot" if label != next(iter(runs)) else "solid",
                        "color": COLORS[method],
                    },
                }
            )
    return traces


def decay_times(cells: dict, method: str, scale: float, filler: int, lengths: list[int]):
    start = cells[method, scale, filler, 0]["state_delta"]
    weight = start.pow(2) / start.pow(2).sum()
    later = [L for L in lengths if L > 0]
    logs = torch.stack(
        [
            torch.log(cells[method, scale, filler, L]["state_delta"] / start.clamp_min(1e-12))
            for L in later
        ]
    )
    x = torch.tensor(later, dtype=torch.float32)[:, None, None]
    slope = (x * logs).sum(0) / (x * x).sum(0)
    tau = torch.where(slope < 0, -1 / slope, torch.full_like(slope, math.inf))
    tau[weight < MIN_SHARE] = math.nan
    return tau, weight


def mechanics(run: Path, lengths: list[int]) -> dict:
    data = torch.load(run / "heads.pt")
    cells, layers = data["cells"], data["layers"]
    keys = sorted({(m, s, f) for m, s, f, _ in cells if m != "clamp"})
    total, taus, cosines = {}, {}, {}
    for method, scale, filler in keys:
        if any((method, scale, filler, L) not in cells for L in lengths):
            continue
        norms = [float(cells[method, scale, filler, L]["state_delta"].norm()) for L in lengths]
        total[method, scale, filler] = [n / norms[0] for n in norms]
        taus[method, scale, filler] = decay_times(cells, method, scale, filler, lengths)
        cosines[method, scale, filler] = [
            cells[method, scale, filler, L]["output_cosine"].mean(1).tolist() for L in lengths
        ]
    return {"layers": layers, "total": total, "taus": taus, "cosines": cosines}


def plot(div: str, traces: list[dict], layout: dict) -> str:
    return (
        f"<div id='{div}' style='height:420px'></div><script>Plotly.newPlot('{div}',"
        f"{json.dumps(traces, allow_nan=True).replace('NaN', 'null').replace('Infinity', 'null')},"
        f"{json.dumps(layout)});</script>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels: dict[str, list[Path]] = {}
    for run in args.runs:
        meta = json.loads((run / "summary.json").read_text())
        labels.setdefault("clean KV" if meta.get("clean_attention") else "normal", []).append(run)
    rows = {
        label: [row for run in runs for row in read_jsonl(run / "rows.jsonl")]
        for label, runs in labels.items()
    }
    for row in (row for group in rows.values() for row in group):
        kind = meta_kind(row["response"])
        row["context_meta"] = float(kind == "context")
        row["garbled_meta"] = float(kind == "garbled")
    lengths = sorted({r["length"] for r in next(iter(rows.values()))})
    feature = meta["feature"]
    log_x = {"type": "log", "title": "filler tokens + 1"}

    sections = [
        f"<p>{escape(feature)}; runs: "
        + ", ".join(
            f"{escape(k)} = {escape(', '.join(str(run) for run in v))}" for k, v in labels.items()
        )
        + ". Solid: normal attention. Dotted: attention keys and values from the unsteered "
        "prompt. Bars: 95% bootstrap over questions.</p>",
        "<h2>Concept rate</h2>"
        + plot(
            "hit",
            behaviour_traces(rows, "hit"),
            {"xaxis": log_x, "yaxis": {"title": "concept rate", "range": [0, 1]}},
        ),
        "<h2>Content quality</h2>"
        + plot(
            "quality",
            behaviour_traces(rows, "content_quality"),
            {"xaxis": log_x, "yaxis": {"title": "quality 0–4", "range": [0, 4]}},
        ),
        "<h2>Responses about the prompt context</h2><p>Regex tag: the answer comments on the "
        "provided or background text (for example, “the text does not contain this "
        "information”) instead of answering.</p>"
        + plot(
            "context",
            behaviour_traces(rows, "context_meta"),
            {"xaxis": log_x, "yaxis": {"title": "share of responses", "range": [0, 1]}},
        ),
        "<h2>Responses that call the question garbled</h2><p>Regex tag: the answer says the "
        "question has typos or is corrupted.</p>"
        + plot(
            "garbled",
            behaviour_traces(rows, "garbled_meta"),
            {"xaxis": log_x, "yaxis": {"title": "share of responses", "range": [0, 1]}},
        ),
    ]

    mech = {}
    for label, runs in labels.items():
        parts = [mechanics(run, lengths) for run in runs]
        mech[label] = {
            "layers": parts[0]["layers"],
            **{
                key: {k: v for part in parts for k, v in part[key].items()}
                for key in ("total", "taus", "cosines")
            },
        }
    total_traces, tau_hist, cosine_traces, heatmaps = [], [], [], []
    for label, data in mech.items():
        dash = "solid" if label == "normal" else "dot"
        for (method, scale, filler), values in data["total"].items():
            name = f"{label}: {method} ×{scale:g}, filler {filler}"
            total_traces.append(
                {
                    "x": [L + 1 for L in lengths],
                    "y": values,
                    "name": name,
                    "mode": "lines+markers",
                    "line": {"dash": dash, "color": COLORS[method]},
                }
            )
            tau, weight = data["taus"][method, scale, filler]
            finite = tau[torch.isfinite(tau)]
            tau_hist.append(
                {"x": finite.tolist(), "type": "histogram", "name": name, "opacity": 0.6}
            )
            heatmaps.append(
                f"<h3>{escape(name)}</h3>"
                + plot(
                    f"tau{len(heatmaps)}",
                    [
                        {
                            "z": tau.clamp(max=4096).tolist(),
                            "y": [str(layer) for layer in data["layers"]],
                            "type": "heatmap",
                            "colorscale": "Viridis",
                            "colorbar": {"title": "τ, tokens"},
                        }
                    ],
                    {"xaxis": {"title": "head"}, "yaxis": {"title": "GDN layer"}},
                )
            )
        for (method, scale, filler), per_length in data["cosines"].items():
            for li, layer in enumerate(data["layers"]):
                cosine_traces.append(
                    {
                        "x": [L + 1 for L in lengths],
                        "y": [values[li] for values in per_length],
                        "name": f"{label}: {method} ×{scale:g} L{layer}",
                        "mode": "lines",
                        "line": {"dash": dash},
                        "visible": "legendonly" if li % 4 else True,
                    }
                )
    sections += [
        "<h2>Total state difference, relative to L = 0</h2>"
        + plot(
            "total",
            total_traces,
            {"xaxis": log_x, "yaxis": {"title": "||ΔS(L)|| / ||ΔS(0)||", "type": "log"}},
        ),
        "<h2>Per-head decay time τ</h2><p>Heads with at least 0.1% of the L = 0 "
        "difference energy. Heads that grow have no τ.</p>"
        + plot(
            "tauhist",
            tau_hist,
            {"barmode": "overlay", "xaxis": {"title": "τ, tokens", "type": "log"}},
        ),
        "<h2>Head output cosine to the unsteered prompt, layer mean</h2>"
        + plot(
            "cosine",
            cosine_traces,
            {"xaxis": log_x, "yaxis": {"title": "cosine"}},
        ),
        "<h2>τ per head</h2>" + "".join(heatmaps),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "<!doctype html><meta charset='utf-8'><title>Filler decay: "
        f"{escape(feature)}</title><script src='{PLOTLY}'></script>"
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:2em auto;"
        "padding:0 1em}</style>" + "".join(sections),
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
