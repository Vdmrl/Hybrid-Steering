"""Plot language-contrast diagnostics collected by ``collect_acts.py``."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.io import to_html
from plotly.subplots import make_subplots


def scalar(head: dict[str, Any], name: str) -> float:
    return float(head["scalars"][name]["mean"])


def effective_rank(matrix: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(matrix)
    p = singular / singular.sum().clamp_min(torch.finfo(singular.dtype).tiny)
    return float((-torch.special.xlogy(p, p).sum()).exp())


def heatmap(figure: go.Figure, values: np.ndarray, layers: list[int], title: str, row: int | None = None, col: int | None = None, *, zmin: float | None = None, zmax: float | None = None, showscale: bool = True) -> None:
    finite = values[np.isfinite(values)]
    low, high = (0.0, 1.0) if not len(finite) else (finite.min(), finite.max())
    if low == high:
        low, high = low - 1, high + 1
    trace = go.Heatmap(x=np.arange(values.shape[1]), y=layers, z=values, colorscale="Viridis", zmin=low if zmin is None else zmin, zmax=high if zmax is None else zmax, showscale=showscale, colorbar={"title": title}, hovertemplate="Layer %{y}<br>Head %{x}<br>" + title + ": %{z:.4g}<extra></extra>")
    if row is None:
        figure.add_trace(trace)
    else:
        figure.add_trace(trace, row=row, col=col)


def arrays(data: dict[str, Any]) -> tuple[list[int], dict[str, np.ndarray]]:
    layers = sorted(data["layers"])
    names = ("||D_mean||_F / E||S||_F", "Effective rank of D_mean", "Mean effective rank of S", "sim_k", "sim_v")
    has_cosine = all("cos_delta_mean" in head["scalars"] for row in data["layers"].values() for head in row)
    values: dict[str, list[list[float]]] = {name: [] for name in names}
    if has_cosine:
        values["Mean cos(D_i, D_mean)"] = []
    for layer in layers:
        for metric in values.values():
            metric.append([])
        for head in data["layers"][layer]:
            if has_cosine:
                values["Mean cos(D_i, D_mean)"][-1].append(scalar(head, "cos_delta_mean"))
            values["||D_mean||_F / E||S||_F"][-1].append(float(torch.linalg.matrix_norm(head["mean_delta"]) / scalar(head, "fro_state")))
            values["Effective rank of D_mean"][-1].append(effective_rank(head["mean_delta"]))
            values["Mean effective rank of S"][-1].append((scalar(head, "effective_rank_a") + scalar(head, "effective_rank_b")) / 2)
            values["sim_k"][-1].append(scalar(head, "sim_k"))
            values["sim_v"][-1].append(scalar(head, "sim_v"))
    return layers, {name: np.array(value) for name, value in values.items()}


def single_heatmap(layers: list[int], values: np.ndarray, title: str) -> go.Figure:
    figure = go.Figure()
    heatmap(figure, values, layers, title)
    figure.update_layout(title=title, height=600, width=780, margin={"l": 70, "r": 120, "t": 70, "b": 65})
    figure.update_xaxes(title="GDN head")
    figure.update_yaxes(title="GDN layer")
    return figure


def factor_heatmaps(layers: list[int], metrics: dict[str, np.ndarray], rank_limit: float | None = None) -> go.Figure:
    mask = False if rank_limit is None else metrics["Mean effective rank of S"] > rank_limit
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Key side: sim_k", "Value side: sim_v"))
    for col, name in enumerate(("sim_k", "sim_v"), 1):
        heatmap(figure, np.where(mask, np.nan, metrics[name]), layers, name, 1, col, zmin=0, zmax=1, showscale=col == 2)
    title = "Leading singular-vector similarity" if rank_limit is None else f"Factor changes where mean effective rank(S) ≤ {rank_limit:g}"
    figure.update_layout(title=title, height=540, margin={"l": 70, "r": 120, "t": 90, "b": 65})
    figure.update_xaxes(title="GDN head")
    figure.update_yaxes(title="GDN layer")
    return figure


def percentile(values: np.ndarray) -> np.ndarray:
    flat = values.ravel()
    return np.searchsorted(np.sort(flat), flat, side="right").reshape(values.shape) / len(flat)


def candidate_map(layers: list[int], metrics: dict[str, np.ndarray]) -> go.Figure:
    score = np.minimum.reduce((percentile(metrics["Mean cos(D_i, D_mean)"]), percentile(metrics["||D_mean||_F / E||S||_F"]), 1 - percentile(metrics["Effective rank of D_mean"])))
    return single_heatmap(layers, score, "Steering candidates: minimum global percentile of consistency, contrast size, and low rank")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a language-steering dashboard.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/steering/language_states.html"))
    parser.add_argument("--low-rank-limit", type=float, default=1.2)
    args = parser.parse_args()
    data = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if not data.get("layers"):
        raise ValueError("artifact has no collected layer statistics")
    layers, metrics = arrays(data)
    figures = []
    if "Mean cos(D_i, D_mean)" in metrics:
        figures.append(("1. Contrast consistency", single_heatmap(layers, metrics["Mean cos(D_i, D_mean)"], "Mean cos(D_i, D_mean)")))
    figures += [
        ("2. Contrast magnitude", single_heatmap(layers, metrics["||D_mean||_F / E||S||_F"], "||D_mean||_F / E||S||_F")),
        ("3. Contrast rank", single_heatmap(layers, metrics["Effective rank of D_mean"], "Effective rank of D_mean")),
        ("4. Singular-vector similarity", factor_heatmaps(layers, metrics)),
        ("5. Low-rank factor changes", factor_heatmaps(layers, metrics, args.low_rank_limit)),
    ]
    if "Mean cos(D_i, D_mean)" in metrics:
        figures.append(("6. Steering candidates", candidate_map(layers, metrics)))
    body = "\n".join(f"<section><h2>{escape(title)}</h2>{to_html(figure, full_html=False, include_plotlyjs=False, config={'responsive': True, 'displaylogo': False})}{'<p>Color = min(percentile(mean cos(D_i, D_mean)), percentile(||D_mean||<sub>F</sub> / E||S||<sub>F</sub>), 1 − percentile(effective rank(D_mean))). Percentiles are over every layer/head cell; brighter cells score well on all three criteria.</p>' if title == '6. Steering candidates' else ''}</section>" for title, figure in figures)
    corpus = data["corpus"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cosine_note = "exact cosine pass collected." if data.get("cosine_collected") else "cosine pass skipped; consistency and candidate plots are unavailable."
    args.output.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Language steering</title><script src='https://cdn.plot.ly/plotly-3.3.0.min.js'></script><style>body{{max-width:1800px;margin:auto;padding:24px;font:15px system-ui;background:#f5f5f5}}section{{margin:24px 0;padding:20px;background:white;border-radius:10px}}h1,h2{{margin-top:0}}</style></head><body><h1>Language steering: {escape(data['model'])}</h1><p>{data['pairs']} {escape(corpus['language_a'])}→{escape(corpus['language_b'])} translations from {escape(corpus['name'])}/{escape(corpus['config'])}; {cosine_note}</p>{body}</body></html>")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
