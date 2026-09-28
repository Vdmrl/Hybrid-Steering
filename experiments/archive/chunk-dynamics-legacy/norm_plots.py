"""Plot state Frobenius norms collected by ``ranks.py``."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
import plotly.graph_objects as go


def write_trends(frame: pd.DataFrame, output: Path) -> None:
    data = frame[(frame.metric == "frobenius_norm") & np.isfinite(frame.value)].groupby(["layer", "position"]).value.mean()
    if data.empty:
        raise ValueError("missing frobenius_norm metric")
    layers = sorted(data.index.get_level_values("layer").unique())
    figure = go.Figure()
    for layer, color in zip(layers, sample_colorscale("Viridis", np.linspace(0, 1, len(layers))), strict=True):
        line = data.loc[layer]
        figure.add_trace(
            go.Scatter(
                x=line.index,
                y=line,
                line={"color": color},
                name=f"layer {layer}",
                legendgroup=str(layer),
                hovertemplate=f"layer={layer}<br>token=%{{x}}<br>Frobenius norm=%{{y:.4g}}<extra></extra>",
            )
        )
    figure.update_layout(
        height=600,
        margin={"r": 150},
        title="Mean state Frobenius norm versus token position by layer",
        legend={"groupclick": "togglegroup"},
        xaxis={"title": "token position", "type": "log"},
        yaxis={"title": "Frobenius norm"},
    )
    figure.write_html(output / "norm_trends.html", include_mathjax="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, help="rank metrics parquet")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.metrics.parent
    output.mkdir(parents=True, exist_ok=True)
    write_trends(pd.read_parquet(args.metrics), output)


if __name__ == "__main__":
    main()
