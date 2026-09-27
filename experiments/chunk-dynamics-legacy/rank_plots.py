"""Plot the rank metrics collected by ``ranks.py``."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
import plotly.graph_objects as go
from plotly.subplots import make_subplots


METRICS = (
    ("effective_rank", "Effective rank"),
    ("stable_rank", "Stable rank"),
    ("r_50", r"$r_{50}$"),
    ("r_90", r"$r_{90}$"),
    ("r_99", r"$r_{99}$"),
)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[np.isfinite(frame.value)]


def quantiles(frame: pd.DataFrame, metric: str, layer: int | None) -> pd.DataFrame:
    data = frame[frame.metric == metric]
    if layer is not None:
        data = data[data.layer == layer]
    return data.groupby("position").value.quantile((0.01, 0.25, 0.5, 0.75, 0.99)).unstack()


def add_boxes(figure: go.Figure, frame: pd.DataFrame, layer: int | None, visible: bool) -> None:
    for index, (metric, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        data = quantiles(frame, metric, layer)
        figure.add_trace(
            go.Box(
                x=data.index,
                q1=data[0.25],
                median=data[0.5],
                q3=data[0.75],
                lowerfence=data[0.01],
                upperfence=data[0.99],
                marker_color="#4C78A8",
                name=label,
                showlegend=False,
                visible=visible,
                hovertemplate=f"token=%{{x}}<br>{label}: %{{median:.4g}}<extra>{label}</extra>",
            ),
            row=row + 1,
            col=column + 1,
        )


def setup_axes(figure: go.Figure) -> None:
    for index, (_, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        figure.update_xaxes(title="token position", type="log", row=row + 1, col=column + 1)
        figure.update_yaxes(title=label, row=row + 1, col=column + 1)


def write_boxes(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=3, cols=2, subplot_titles=[label for _, label in METRICS])
    add_boxes(figure, frame, None, True)
    setup_axes(figure)
    figure.update_layout(height=1000, showlegend=False, title="State rank versus token position (all layers and heads)")
    figure.write_html(output / "rank_metrics.html", include_mathjax="cdn")


def write_trends(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=3, cols=2, subplot_titles=[label for _, label in METRICS])
    layers = sorted(frame.layer.unique())
    colors = sample_colorscale("Viridis", np.linspace(0, 1, len(layers)))
    for index, (metric, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        data = frame[frame.metric == metric].groupby(["layer", "position"]).value.mean()
        for layer, color in zip(layers, colors, strict=True):
            line = data.loc[layer]
            figure.add_trace(
                go.Scatter(
                    x=line.index,
                    y=line,
                    line={"color": color},
                    name=f"layer {layer}",
                    legendgroup=str(layer),
                    showlegend=index == 0,
                    hovertemplate=f"layer={layer}<br>token=%{{x}}<br>{label}=%{{y:.4g}}<extra></extra>",
                ),
                row=row + 1,
                col=column + 1,
            )
        figure.update_xaxes(title="token position", type="log", row=row + 1, col=column + 1)
        figure.update_yaxes(title=label, row=row + 1, col=column + 1)
    figure.update_layout(height=1000, margin={"r": 150}, title="Mean state rank versus token position by layer", legend={"groupclick": "togglegroup"})
    figure.write_html(output / "rank_trends.html", include_mathjax="cdn")


def write_layer_boxes(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=3, cols=2, subplot_titles=[label for _, label in METRICS])
    layers = sorted(frame.layer.unique())
    for layer in layers:
        add_boxes(figure, frame, layer, layer == layers[0])
    setup_axes(figure)
    figure.update_layout(
        height=1000,
        showlegend=False,
        title="State rank versus token position by layer",
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": f"layer {layer}",
                        "method": "update",
                        "args": [{"visible": [selected == layer for selected in layers for _ in METRICS]}],
                    }
                    for layer in layers
                ],
                "x": 0,
                "y": 1.08,
            }
        ],
    )
    figure.write_html(output / "rank_metrics_by_layer.html", include_mathjax="cdn")


def write_heatmaps(frame: pd.DataFrame, output: Path) -> None:
    rows = (len(METRICS) + 1) // 2
    figure = make_subplots(rows=rows, cols=2, subplot_titles=[label for _, label in METRICS], vertical_spacing=0.11, horizontal_spacing=0.16)
    for index, (metric, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        data = frame[frame.metric == metric].groupby(["head", "layer"]).value.mean().unstack()
        figure.add_trace(
            go.Heatmap(
                x=data.columns,
                y=data.index,
                z=data.values,
                colorscale="Viridis",
                colorbar={"title": label, "x": 0.44 if column == 0 else 1.02, "len": 0.22, "y": 1 - (row + 0.5) / rows},
                hovertemplate="layer=%{x}<br>head=%{y}<br>mean=" + label + "=%{z:.4g}<extra></extra>",
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.update_xaxes(title="decoder layer", row=row + 1, col=column + 1)
        figure.update_yaxes(title="head", row=row + 1, col=column + 1)
    figure.update_layout(height=1050, margin={"r": 150, "t": 80}, title="Mean state rank by layer and head")
    figure.write_html(output / "rank_heatmaps.html", include_mathjax="cdn")


def write_plots(frame: pd.DataFrame, output: Path) -> None:
    frame = clean(frame)
    write_boxes(frame, output)
    write_trends(frame, output)
    write_layer_boxes(frame, output)
    write_heatmaps(frame, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, help="rank metrics parquet")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.metrics.parent
    output.mkdir(parents=True, exist_ok=True)
    write_plots(pd.read_parquet(args.metrics), output)


if __name__ == "__main__":
    main()
