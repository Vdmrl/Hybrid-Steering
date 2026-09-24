"""Build interactive HTML dashboards from sharded GDN metric Parquet files."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import duckdb
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.io import to_html
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

METRICS = {
    "effective_rank": "Effective rank",
    "energy_rank_50": "Energy rank (50%)",
    "energy_rank_90": "Energy rank (90%)",
    "energy_rank_99": "Energy rank (99%)",
    "stable_rank": "Stable rank",
    "frobenius_norm": "Frobenius norm",
    "sigma_1": "Largest singular value",
    "anisotropy": "Top-component energy fraction",
    "relative_update": "Relative state update",
    "state_cosine": "Cosine similarity to previous saved state",
    "subspace_overlap_8": "Top-8 right-subspace overlap",
}
TYPES = ("coherent", "shuffled", "repeated")
COLORS = {"coherent": "#2166ac", "shuffled": "#d6604d", "repeated": "#1b9e77"}
PLOT_CONFIG = {"displaylogo": False, "responsive": True, "scrollZoom": True}
SUMMARY_POSITIONS = (16, 64, 256, 1024, 4096)


def fixed_positions(available: np.ndarray) -> list[int]:
    available_set = set(available)
    maximum = int(max(available_set))
    return [
        position
        for position in sorted({*SUMMARY_POSITIONS, maximum})
        if position in available_set
    ]


def color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        padding = abs(low) * 0.01 or 1.0
        return low - padding, high + padding
    return low, high


def figure_html(figure: go.Figure) -> str:
    return to_html(
        figure,
        full_html=False,
        include_plotlyjs=False,
        config=PLOT_CONFIG,
    )


def write_page(
    path: Path,
    title: str,
    figures: list[tuple[str, go.Figure]],
    root: str = ".",
) -> None:
    sections = "\n".join(
        f"<section><h2>{escape(heading)}</h2>{figure_html(figure)}</section>"
        for heading, figure in figures
    )
    head_options = "\n".join(
        f'<option value="{root}/heads/{metric}.html">{escape(label)}</option>'
        for metric, label in METRICS.items()
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <script src="{root}/plotly.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #1f2937; font: 15px/1.45 system-ui, sans-serif;
            background: #f3f5f8; }}
    main {{ max-width: 1800px; margin: 0 auto; padding: 28px; }}
    header {{ display: flex; flex-wrap: wrap; align-items: center; gap: 16px;
              justify-content: space-between; margin-bottom: 22px; }}
    h1 {{ margin: 0; font-size: 26px; font-weight: 650; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; font-weight: 620; }}
    nav {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }}
    nav a {{ color: #155e75; font-weight: 600; text-decoration: none; }}
    select {{ padding: 6px 9px; border: 1px solid #cbd5e1; border-radius: 6px;
              background: white; font: inherit; }}
    section {{ margin: 22px 0; padding: 20px; overflow-x: auto; background: white;
               border: 1px solid #dbe1e8; border-radius: 12px;
               box-shadow: 0 1px 2px #0f172a0d; }}
    .plotly-graph-div {{ min-width: 860px; }}
    @media (max-width: 920px) {{
      main {{ padding: 14px; }}
      section {{ padding: 14px; }}
      .plotly-graph-div {{ min-width: 760px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{escape(title)}</h1>
      <nav><a href="{root}/dashboard.html">Main dashboard</a>
      <label>All-head heatmap:
        <select onchange="if (this.value) location.href = this.value">
          <option value="">Choose a metric</option>
          {head_options}
        </select>
      </label></nav>
    </header>
    {sections}
  </main>
</body>
</html>
"""
    )


def aggregate_metrics(connection, parquet: str):
    columns = ", ".join(f"median({metric}) AS {metric}" for metric in METRICS)
    return connection.execute(
        f"""
        WITH heads AS (
          SELECT data_type, layer, head, position, {columns}
          FROM read_parquet('{parquet}')
          GROUP BY data_type, layer, head, position
        )
        SELECT data_type, layer, position,
               {
            ", ".join(
                f"median({metric}) AS {metric}, "
                f"quantile_cont({metric}, 0.25) AS {metric}_q25, "
                f"quantile_cont({metric}, 0.75) AS {metric}_q75"
                for metric in METRICS
            )
        }
        FROM heads
        GROUP BY data_type, layer, position
        ORDER BY data_type, layer, position
        """
    ).fetchdf()


def phase_figure(frame) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=TYPES,
        vertical_spacing=0.06,
    )
    limits = {}
    for metric_index, (metric, label) in enumerate(METRICS.items()):
        limits[metric] = {
            data_type: color_limits(
                frame.loc[frame.data_type == data_type, metric].to_numpy()
            )
            for data_type in TYPES
        }
        for row, data_type in enumerate(TYPES, 1):
            pivot = frame[frame.data_type == data_type].pivot(
                index="layer", columns="position", values=metric
            )
            low, high = limits[metric][data_type]
            figure.add_trace(
                go.Heatmap(
                    x=pivot.columns,
                    y=pivot.index,
                    z=pivot.to_numpy(),
                    colorscale="Viridis",
                    zmin=low,
                    zmax=high,
                    colorbar={
                        "title": f"{data_type}<br>local scale",
                        "x": 1.02,
                        "y": 1 - (row - 0.5) / len(TYPES),
                        "len": 0.26,
                    },
                    visible=metric_index == 0,
                    hovertemplate=(
                        "Token %{x}<br>Layer %{y}<br>"
                        + label
                        + ": %{z:.5g}<extra>"
                        + data_type
                        + "</extra>"
                    ),
                ),
                row=row,
                col=1,
            )
    buttons = []
    traces_per_metric = len(TYPES)
    for metric_index, (metric, label) in enumerate(METRICS.items()):
        visible = [False] * len(figure.data)
        start = metric_index * traces_per_metric
        visible[start : start + traces_per_metric] = [True] * traces_per_metric
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"{label}: layer-token phase diagram"},
                ],
            }
        )
    first_metric = next(iter(METRICS))
    figure.update_layout(
        title=f"{METRICS[first_metric]}: layer-token phase diagram",
        height=1080,
        margin={"l": 70, "r": 170, "t": 125, "b": 60},
        updatemenus=[{"buttons": buttons, "x": 0, "y": 1.07, "xanchor": "left"}],
    )
    figure.update_yaxes(title_text="GDN layer")
    figure.update_xaxes(title_text="Token position", row=3, col=1)
    return figure


def layer_slices_figure(frame, positions: list[int]) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=TYPES,
    )
    position_colors = sample_colorscale("Viridis", np.linspace(0, 1, len(positions)))
    traces_per_metric = len(TYPES) * len(positions)
    for metric_index, (metric, label) in enumerate(METRICS.items()):
        for column, data_type in enumerate(TYPES, 1):
            for position, color in zip(positions, position_colors, strict=True):
                part = frame[
                    (frame.data_type == data_type) & (frame.position == position)
                ].sort_values("layer")
                figure.add_trace(
                    go.Scatter(
                        x=part.layer,
                        y=part[metric],
                        mode="lines+markers",
                        name=str(position),
                        legendgroup=str(position),
                        showlegend=column == 1,
                        line={"color": color},
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": part[f"{metric}_q75"] - part[metric],
                            "arrayminus": part[metric] - part[f"{metric}_q25"],
                            "thickness": 0.8,
                        },
                        visible=metric_index == 0,
                        hovertemplate=(
                            "Layer %{x}<br>Median head: %{y:.5g}"
                            "<extra>token " + str(position) + "</extra>"
                        ),
                    ),
                    row=1,
                    col=column,
                )
    buttons = []
    for metric_index, label in enumerate(METRICS.values()):
        visible = [False] * len(figure.data)
        start = metric_index * traces_per_metric
        visible[start : start + traces_per_metric] = [True] * traces_per_metric
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"{label} across layers at fixed token positions",
                        "yaxis": {"title": label},
                    },
                ],
            }
        )
    first_label = next(iter(METRICS.values()))
    figure.update_layout(
        title=f"{first_label} across layers at fixed token positions",
        height=550,
        margin={"l": 75, "r": 35, "t": 115, "b": 65},
        legend_title="Token position",
        updatemenus=[{"buttons": buttons, "x": 0, "y": 1.08, "xanchor": "left"}],
    )
    figure.update_xaxes(title_text="GDN layer")
    figure.update_yaxes(title_text=first_label, row=1, col=1)
    return figure


def counts_figure(counts) -> go.Figure:
    figure = go.Figure()
    for data_type in TYPES:
        part = counts[counts.data_type == data_type]
        figure.add_scatter(
            x=part.position,
            y=part.available_count,
            mode="lines",
            name=data_type,
            line={"color": COLORS[data_type]},
            hovertemplate=(
                f"Token %{{x}}<br>Sequences %{{y}}<extra>{data_type}</extra>"
            ),
        )
    figure.update_layout(
        height=420,
        xaxis_title="Token position",
        yaxis_title="Available sequences",
    )
    return figure


def spectrum_figure(frame) -> go.Figure:
    layers = sorted(frame.layer.unique())
    positions = sorted(frame.position.unique())
    colors = sample_colorscale("Viridis", np.linspace(0, 1, len(positions)))
    figure = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=TYPES,
    )
    traces_per_layer = len(TYPES) * len(positions)
    for layer_index, layer in enumerate(layers):
        for column, data_type in enumerate(TYPES, 1):
            for position, color in zip(positions, colors, strict=True):
                part = frame[
                    (frame.data_type == data_type)
                    & (frame.layer == layer)
                    & (frame.position == position)
                ].sort_values("spectrum_index")
                figure.add_scatter(
                    x=part.spectrum_index,
                    y=part.metric_value,
                    mode="lines+markers",
                    name=str(position),
                    legendgroup=str(position),
                    showlegend=column == 1 and layer_index == 0,
                    line={"color": color},
                    visible=layer_index == 0,
                    hovertemplate=(
                        "Index %{x}<br>Energy fraction %{y:.5g}"
                        "<extra>token " + str(position) + "</extra>"
                    ),
                    row=1,
                    col=column,
                )
    layer_buttons = []
    for layer_index, layer in enumerate(layers):
        visible = [False] * len(figure.data)
        start = layer_index * traces_per_layer
        visible[start : start + traces_per_layer] = [True] * traces_per_layer
        layer_buttons.append(
            {
                "label": f"Layer {layer}",
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Normalized singular spectrum: GDN layer {layer}"},
                ],
            }
        )
    figure.update_layout(
        title=f"Normalized singular spectrum: GDN layer {layers[0]}",
        height=550,
        margin={"l": 75, "r": 35, "t": 115, "b": 65},
        legend_title="Token position",
        updatemenus=[
            {"buttons": layer_buttons, "x": 0, "y": 1.08, "xanchor": "left"},
            {
                "buttons": [
                    {
                        "label": "Linear Y",
                        "method": "relayout",
                        "args": [
                            {
                                "yaxis.type": "linear",
                                "yaxis2.type": "linear",
                                "yaxis3.type": "linear",
                            }
                        ],
                    },
                    {
                        "label": "Log Y",
                        "method": "relayout",
                        "args": [
                            {
                                "yaxis.type": "log",
                                "yaxis2.type": "log",
                                "yaxis3.type": "log",
                            }
                        ],
                    },
                ],
                "x": 0.22,
                "y": 1.08,
                "xanchor": "left",
            },
        ],
    )
    figure.update_xaxes(title_text="Singular-value index")
    figure.update_yaxes(title_text="Fraction of spectral energy", row=1, col=1)
    return figure


def write_head_pages(connection, parquet: str, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    for metric, label in METRICS.items():
        frame = connection.execute(
            f"""
            SELECT data_type, layer, head, position,
                   median({metric}) AS metric_value
            FROM read_parquet('{parquet}')
            GROUP BY data_type, layer, head, position
            ORDER BY data_type, layer, head, position
            """
        ).fetchdf()
        layers = sorted(frame.layer.unique())
        limits = {
            data_type: color_limits(
                frame.loc[frame.data_type == data_type, "metric_value"].to_numpy()
            )
            for data_type in TYPES
        }
        figure = make_subplots(
            rows=1,
            cols=3,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=TYPES,
        )
        for layer_index, layer in enumerate(layers):
            for column, data_type in enumerate(TYPES, 1):
                pivot = frame[
                    (frame.layer == layer) & (frame.data_type == data_type)
                ].pivot(index="head", columns="position", values="metric_value")
                low, high = limits[data_type]
                figure.add_trace(
                    go.Heatmap(
                        x=pivot.columns,
                        y=pivot.index,
                        z=pivot.to_numpy(),
                        colorscale="Turbo",
                        zmin=low,
                        zmax=high,
                        colorbar={
                            "title": f"{data_type}<br>local scale",
                            "x": 0.295 + (column - 1) * 0.355,
                            "y": 0.5,
                            "len": 0.84,
                        },
                        visible=layer_index == 0,
                        hovertemplate=(
                            "Token %{x}<br>Head %{y}<br>"
                            + label
                            + ": %{z:.5g}<extra>"
                            + data_type
                            + "</extra>"
                        ),
                    ),
                    row=1,
                    col=column,
                )
        buttons = []
        for layer_index, layer in enumerate(layers):
            visible = [False] * len(figure.data)
            start = layer_index * len(TYPES)
            visible[start : start + len(TYPES)] = [True] * len(TYPES)
            buttons.append(
                {
                    "label": f"Layer {layer}",
                    "method": "update",
                    "args": [
                        {"visible": visible},
                        {"title": f"{label}: all heads, GDN layer {layer}"},
                    ],
                }
            )
        figure.update_layout(
            title=f"{label}: all heads, GDN layer {layers[0]}",
            height=680,
            margin={"l": 75, "r": 65, "t": 115, "b": 65},
            updatemenus=[{"buttons": buttons, "x": 0, "y": 1.08, "xanchor": "left"}],
        )
        figure.update_xaxes(title_text="Token position")
        figure.update_yaxes(title_text="Value head", row=1, col=1)
        page = output_dir / f"{metric}.html"
        write_page(
            page,
            f"{label}: all-head heatmaps",
            [("Head dynamics", figure)],
            "..",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build interactive GDN state-dynamics dashboards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="timestamped artifacts/state_dynamics run directory",
    )
    args = parser.parse_args()
    input_dir = args.run_dir / "intermediate" / "state"
    output_dir = args.run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plotly.min.js").write_text(get_plotlyjs())

    parquet = str(input_dir / "metrics-*.parquet")
    connection = duckdb.connect()
    counts = connection.execute(
        f"""
        SELECT data_type, position, count(DISTINCT text_id) AS available_count
        FROM read_parquet('{parquet}')
        GROUP BY data_type, position ORDER BY data_type, position
        """
    ).fetchdf()
    counts.to_csv(output_dir / "available_sequences.csv", index=False)
    aggregate = aggregate_metrics(connection, parquet)
    positions = fixed_positions(aggregate.position.unique())
    spectrum = connection.execute(
        f"""
        WITH heads AS (
          SELECT data_type, layer, head, position, ordinal AS spectrum_index,
                 median(spectrum_value) AS metric_value
          FROM (
            SELECT data_type, layer, head, position, singular_energy_normalized
            FROM read_parquet('{parquet}')
            WHERE position IN ({",".join(map(str, positions))})
          ), UNNEST(singular_energy_normalized)
             WITH ORDINALITY AS u(spectrum_value, ordinal)
          GROUP BY data_type, layer, head, position, ordinal
        )
        SELECT data_type, layer, position, spectrum_index,
               median(metric_value) AS metric_value
        FROM heads
        GROUP BY data_type, layer, position, spectrum_index
        ORDER BY data_type, layer, position, spectrum_index
        """
    ).fetchdf()
    write_page(
        output_dir / "dashboard.html",
        "GDN state dynamics",
        [
            ("Layer-token phase diagrams", phase_figure(aggregate)),
            (
                "Layer slices at fixed token positions",
                layer_slices_figure(aggregate, positions),
            ),
            ("Available sequences", counts_figure(counts)),
            ("Normalized singular spectra", spectrum_figure(spectrum)),
        ],
    )
    write_head_pages(connection, parquet, output_dir / "heads")
    print(f"wrote interactive dashboards to {output_dir}")


if __name__ == "__main__":
    main()
