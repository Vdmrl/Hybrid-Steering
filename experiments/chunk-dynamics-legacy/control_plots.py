"""Plot the value-similarity controls collected by ``controls.py``."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
import plotly.graph_objects as go
from plotly.subplots import make_subplots


METRICS = (
    ("sim_v", "Next-token value cosine similarity"),
    ("sim_v_mean", "Dataset-mean value cosine similarity"),
    ("sim_v_centered", "Mean-centered value cosine similarity"),
    ("sim_v_global_axis", "Global value-axis cosine similarity"),
    ("sim_v_random", "Random value cosine similarity"),
    ("sim_v_to_random_v", "Random value–value cosine similarity"),
)
KEYS = ["document", "layer", "position", "head"]
ROWS = (len(METRICS) + 1) // 2


def metric_frame(first: Path, global_: Path) -> pd.DataFrame:
    first_frame = pd.read_parquet(first)
    global_frame = pd.read_parquet(global_)
    names = {name for name, _ in METRICS}
    frame = pd.concat((first_frame, global_frame), ignore_index=True)
    frame = frame[frame.metric.isin(names) & np.isfinite(frame.value)]
    missing = names - set(frame.metric.unique())
    if missing:
        raise ValueError(f"missing metrics: {', '.join(sorted(missing))}")
    return frame


def write_heatmaps(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=ROWS, cols=2, subplot_titles=[label for _, label in METRICS], vertical_spacing=0.11, horizontal_spacing=0.16)
    for index, (metric, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        data = frame[frame.metric == metric].groupby(["head", "layer"]).value.mean().unstack()
        figure.add_trace(go.Heatmap(x=data.columns, y=data.index, z=data.values, colorscale="Viridis", colorbar={"title": label, "x": 0.44 if column == 0 else 1.02, "len": 0.22, "y": 1 - (row + 0.5) / ROWS}), row=row + 1, col=column + 1)
        figure.update_xaxes(title="decoder layer", row=row + 1, col=column + 1)
        figure.update_yaxes(title="head", row=row + 1, col=column + 1)
    figure.update_layout(height=1050, margin={"r": 150, "t": 80}, title="Mean similarity by head and layer")
    figure.write_html(output / "control_head_layer_heatmaps.html", include_mathjax="cdn")


def write_distributions(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=ROWS, cols=2, subplot_titles=[label for _, label in METRICS], vertical_spacing=0.11, horizontal_spacing=0.12)
    for index, (metric, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        values = frame.loc[frame.metric == metric, "value"].to_numpy()
        low, high = np.quantile(values, (0.001, 0.999))
        counts, edges = np.histogram(values[(values >= low) & (values <= high)], bins=150, range=(low, high), density=True)
        figure.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, marker_color="#4C78A8", hovertemplate=f"{label}=%{{x:.4g}}<br>density=%{{y:.4g}}<extra></extra>"), row=row + 1, col=column + 1)
        figure.update_xaxes(title=label, row=row + 1, col=column + 1)
        figure.update_yaxes(title="density", row=row + 1, col=column + 1)
    figure.update_layout(height=1050, showlegend=False, title="Similarity distributions (central 99.8%)")
    figure.write_html(output / "control_distributions.html", include_mathjax="cdn")


def write_differences(frame: pd.DataFrame, output: Path) -> None:
    wide = frame.pivot_table(index=KEYS, columns="metric", values="value")
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Next-token minus global-axis cosine similarity", "Next-token minus random-value cosine similarity"), horizontal_spacing=0.16)
    for column, baseline in enumerate(("sim_v_global_axis", "sim_v_random"), 1):
        data = (wide.sim_v - wide[baseline]).groupby([wide.index.get_level_values("head"), wide.index.get_level_values("layer")]).mean().unstack()
        bound = np.quantile(np.abs(data.to_numpy()), 0.99)
        figure.add_trace(go.Heatmap(x=data.columns, y=data.index, z=data.values, colorscale="RdBu", zmid=0, zmin=-bound, zmax=bound, colorbar={"title": "difference", "x": 0.44 if column == 1 else 1.02}), row=1, col=column)
        figure.update_xaxes(title="decoder layer", row=1, col=column)
        figure.update_yaxes(title="head", row=1, col=column)
    figure.update_layout(height=450, margin={"r": 150, "t": 80}, title="Mean local cosine-similarity advantage")
    figure.write_html(output / "control_difference_heatmaps.html", include_mathjax="cdn")


def write_paired_scatter(frame: pd.DataFrame, output: Path) -> None:
    data = frame.pivot_table(index=KEYS, columns="metric", values="value", aggfunc="first").dropna(subset=["sim_v_global_axis", "sim_v"])
    data = data.sample(min(len(data), 200_000), random_state=0)
    layers = sorted(data.index.get_level_values("layer").unique())
    colors = sample_colorscale("Viridis", np.linspace(0, 1, len(layers)))
    figure = go.Figure()
    for layer, color in zip(layers, colors, strict=True):
        points = data.xs(layer, level="layer")
        figure.add_trace(go.Scattergl(x=points.sim_v_global_axis, y=points.sim_v, mode="markers", name=f"layer {layer}", marker={"color": color, "size": 4, "opacity": 0.45}, customdata=points.index.to_frame(index=False), hovertemplate="document=%{customdata[0]}<br>position=%{customdata[1]}<br>head=%{customdata[2]}<br>global axis=%{x:.4g}<br>next-token=%{y:.4g}<extra></extra>"))
    figure.update_layout(height=650, margin={"r": 30, "t": 80}, title="Next-token versus global-axis value cosine similarity", xaxis_title="sim_v_global_axis", yaxis_title="sim_v")
    figure.write_html(output / "control_global_axis_vs_sim_v.html", include_mathjax="cdn")


def write_position(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=ROWS, cols=2, subplot_titles=[label for _, label in METRICS], vertical_spacing=0.11, horizontal_spacing=0.12)
    layers = sorted(frame.layer.unique())
    colors = sample_colorscale("Viridis", np.linspace(0, 1, len(layers)))
    for index, (metric, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        data = frame[frame.metric == metric].groupby(["layer", "position"]).value.mean()
        for layer, color in zip(layers, colors, strict=True):
            line = data.loc[layer]
            figure.add_trace(go.Scatter(x=line.index, y=line, line={"color": color}, name=f"layer {layer}", legendgroup=str(layer), showlegend=index == 0, hovertemplate=f"layer={layer}<br>token=%{{x}}<br>{label}=%{{y:.4g}}<extra></extra>"), row=row + 1, col=column + 1)
        figure.update_xaxes(title="token position", type="log", row=row + 1, col=column + 1)
        figure.update_yaxes(title=label, row=row + 1, col=column + 1)
    figure.update_layout(height=1050, margin={"r": 150, "t": 80}, title="Mean similarity versus token position by layer", legend={"groupclick": "togglegroup"})
    figure.write_html(output / "control_similarity_by_position.html", include_mathjax="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path, help="first-pass parquet (local/random controls)")
    parser.add_argument("global_", type=Path, help="global-pass parquet (centered/global-axis controls)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.first.parent
    output.mkdir(parents=True, exist_ok=True)
    frame = metric_frame(args.first, args.global_)
    write_heatmaps(frame, output)
    write_distributions(frame, output)
    write_differences(frame, output)
    write_paired_scatter(frame, output)
    write_position(frame, output)


if __name__ == "__main__":
    main()
