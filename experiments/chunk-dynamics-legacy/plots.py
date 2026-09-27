from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRICS = (
    ("sv1", r"$\sigma_1$"),
    ("sv_ratio", r"$\sigma_2 / \sigma_1$"),
    ("sim_v", r"$|u^T v| / \|v\|$"),
    ("sim_k", r"$|r^T k| / (\|r\|\|k\|)$"),
    ("log_rho", r"$\log \rho$"),
    ("key_retain", r"$\|(I-\beta kk^T)r\|$"),
)
HEATMAPS = METRICS[2:5]


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[np.isfinite(frame.value)]


def box_data(frame: pd.DataFrame, metric: str, layer: int | None) -> pd.DataFrame:
    data = frame[frame.metric == metric]
    if layer is not None:
        data = data[data.layer == layer]
    return data.groupby("position").value.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).unstack()


def add_boxes(figure: go.Figure, frame: pd.DataFrame, layer: int | None, visible: bool) -> None:
    for index, (metric, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        data = box_data(frame, metric, layer)
        figure.add_trace(
            go.Box(
                x=data.index.astype(str),
                q1=data[0.25],
                median=data[0.5],
                q3=data[0.75],
                lowerfence=data[0.01],
                upperfence=data[0.99],
                name=label,
                showlegend=False,
                visible=visible,
                marker_color="#4C78A8",
                hovertemplate=f"token=%{{x}}<br>{label}: %{{median:.4g}}<extra>{label}</extra>",
            ),
            row=row + 1,
            col=column + 1,
        )


def box_figure(frame: pd.DataFrame, layer: int | None = None) -> go.Figure:
    figure = make_subplots(rows=3, cols=2, subplot_titles=[label for _, label in METRICS])
    add_boxes(figure, frame, layer, True)
    positions = [str(value) for value in sorted(frame.position.unique())]
    figure.update_xaxes(title="token position", categoryorder="array", categoryarray=positions)
    for index, (metric, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        values = frame[frame.metric == metric]
        if layer is not None:
            values = values[values.layer == layer]
        values = values.value
        figure.update_yaxes(title=label, range=values.quantile([0.01, 0.99]).tolist(), row=row + 1, col=column + 1)
    figure.update_layout(showlegend=False, height=1000)
    return figure


def write_layer_boxes(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=3, cols=2, subplot_titles=[label for _, label in METRICS])
    layers = sorted(frame.layer.unique())
    for layer in layers:
        add_boxes(figure, frame, layer, layer == layers[0])
    positions = [str(value) for value in sorted(frame.position.unique())]
    figure.update_xaxes(title="token position", categoryorder="array", categoryarray=positions)
    for index, (_, label) in enumerate(METRICS):
        row, column = divmod(index, 2)
        figure.update_yaxes(title=label, row=row + 1, col=column + 1)
    traces_per_layer = len(METRICS)
    figure.update_layout(
        height=1000,
        showlegend=False,
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": f"layer {layer}",
                        "method": "update",
                        "args": [{"visible": [item == layer for item in layers for _ in METRICS]}],
                    }
                    for layer in layers
                ],
                "x": 0,
                "y": 1.08,
            }
        ],
    )
    figure.write_html(output / "state_metrics_by_layer.html", include_mathjax="cdn")


def write_heatmaps(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=2, cols=2, subplot_titles=[label for _, label in HEATMAPS], vertical_spacing=0.14, horizontal_spacing=0.16)
    for index, (metric, label) in enumerate(HEATMAPS):
        row, column = divmod(index, 2)
        data = frame[frame.metric == metric].groupby(["head", "layer"]).value.mean().unstack()
        figure.add_trace(
            go.Heatmap(
                x=data.columns,
                y=data.index,
                z=data.values,
                colorscale="Viridis",
                colorbar={"title": label, "x": 0.44 if column == 0 else 1.02, "len": 0.32, "y": 1 - (row + 0.5) / 2},
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.update_xaxes(title="decoder layer", row=row + 1, col=column + 1)
        figure.update_yaxes(title="head", row=row + 1, col=column + 1)
    figure.update_layout(height=750, margin={"r": 150, "t": 80})
    figure.write_html(output / "state_heatmaps.html", include_mathjax="cdn")


def write_similarity_scatter(frame: pd.DataFrame, output: Path) -> None:
    data = (
        frame[frame.metric.isin(("sim_v", "sim_k"))]
        .pivot_table(index=["document", "layer", "position", "head"], columns="metric", values="value")
        .reset_index()
    )
    data = data.sample(min(len(data), 100_000), random_state=0)
    figure = make_subplots(
        rows=2, cols=2, subplot_titles=("uniform", "token position", "decoder layer"), vertical_spacing=0.16, horizontal_spacing=0.16
    )
    for index, color, title in ((0, "#4C78A8", None), (1, data.position, "token position"), (2, data.layer, "decoder layer")):
        row, column = divmod(index, 2)
        marker = {"size": 3, "opacity": 0.3, "color": color}
        if title is not None:
            marker.update(
                colorscale="Viridis", colorbar={"title": title, "x": 0.44 if column == 0 else 1.02, "len": 0.32, "y": 1 - (row + 0.5) / 2}
            )
        figure.add_trace(
            go.Scattergl(
                x=data.sim_v,
                y=data.sim_k,
                customdata=data[["document", "layer", "position", "head"]],
                mode="markers",
                marker=marker,
                name=title or "similarity",
                showlegend=False,
                hovertemplate=r"$|u^T v| / \|v\|$=%{x:.4g}<br>$|r^T k| / (\|r\|\|k\|)$=%{y:.4g}<br>document=%{customdata[0]}<br>layer=%{customdata[1]}<br>token=%{customdata[2]}<br>head=%{customdata[3]}<extra>similarity</extra>",
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.update_xaxes(title=r"$|u^T v| / \|v\|$", range=[0, 1], row=row + 1, col=column + 1)
        figure.update_yaxes(
            title=r"$|r^T k| / (\|r\|\|k\|)$",
            range=[0, 1],
            row=row + 1,
            col=column + 1,
        )
    figure.update_layout(height=800, margin={"r": 150, "t": 80})
    figure.write_html(output / "sim_v_vs_sim_k.html", include_mathjax="cdn")


def write_similarity_density(frame: pd.DataFrame, output: Path) -> None:
    data = frame[frame.metric.isin(("sim_v", "sim_k"))].pivot_table(
        index=["document", "layer", "position", "head"], columns="metric", values="value"
    )
    x_low, x_high = data.sim_v.quantile([0.001, 0.999])
    y_low, y_high = data.sim_k.quantile([0.001, 0.999])
    counts, x_edges, y_edges = np.histogram2d(data.sim_v, data.sim_k, bins=250, range=((x_low, x_high), (y_low, y_high)))
    sim_v_counts, sim_v_edges = np.histogram(data.sim_v, bins=150, range=(x_low, x_high), density=True)
    sim_k_counts, sim_k_edges = np.histogram(data.sim_k, bins=150, range=(y_low, y_high), density=True)
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, None], [{}, {}]],
        column_widths=[0.78, 0.22],
        row_heights=[0.22, 0.78],
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
    )
    figure.add_trace(
        go.Heatmap(
            x=(x_edges[:-1] + x_edges[1:]) / 2,
            y=(y_edges[:-1] + y_edges[1:]) / 2,
            z=np.log1p(counts).T,
            colorscale="Viridis",
            zsmooth="best",
            colorbar={"title": "log(1 + count)", "x": 1.18},
            hovertemplate=r"$|u^T v| / \|v\|$=%{x:.4g}<br>$|r^T k| / (\|r\|\|k\|)$=%{y:.4g}<br>log(1 + count)=%{z:.3g}<extra>density</extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=(sim_v_edges[:-1] + sim_v_edges[1:]) / 2,
            y=sim_v_counts,
            marker_color="#4C78A8",
            hovertemplate=r"$|u^T v| / \|v\|$=%{x:.4g}<br>density=%{y:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=sim_k_counts,
            y=(sim_k_edges[:-1] + sim_k_edges[1:]) / 2,
            orientation="h",
            marker_color="#4C78A8",
            hovertemplate=r"$|r^T k| / (\|r\|\|k\|)$=%{y:.4g}<br>density=%{x:.4g}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(showticklabels=False, range=[x_low, x_high], row=1, col=1)
    figure.update_yaxes(title="density", row=1, col=1)
    figure.update_xaxes(title="density", row=2, col=2)
    figure.update_yaxes(showticklabels=False, range=[y_low, y_high], row=2, col=2)
    figure.update_xaxes(title=r"$|u^T v| / \|v\|$", range=[x_low, x_high], row=2, col=1)
    figure.update_yaxes(title=r"$|r^T k| / (\|r\|\|k\|)$", range=[y_low, y_high], row=2, col=1)
    figure.update_layout(height=850, showlegend=False, margin={"r": 150, "t": 80})
    figure.write_html(output / "sim_v_vs_sim_k_density.html", include_mathjax="cdn")


def write_plots(frame: pd.DataFrame, output: Path) -> None:
    frame = clean(frame)
    box_figure(frame).write_html(output / "state_metrics.html", include_mathjax="cdn")
    write_layer_boxes(frame, output)
    write_heatmaps(frame, output)
    write_similarity_scatter(frame, output)
    write_similarity_density(frame, output)
