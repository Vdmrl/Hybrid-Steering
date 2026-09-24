"""Build one HTML report for a forgetting scale sweep."""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_metrics(directory: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("scale_*/prefix_*.jsonl")):
        scale = float(path.parent.name.removeprefix("scale_").replace("_", "."))
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                response_rows.append({**row, "scale": scale})
                if row["condition"] != "steered":
                    continue
                for layer, value in row["output_delta_norm"].items():
                    metric_rows.append({"scale": scale, "prefix_length": row["prefix_length"], "layer": int(layer), "metric": "delta output norm", "value": value})
                for layer, value in row["output_cosine"].items():
                    metric_rows.append({"scale": scale, "prefix_length": row["prefix_length"], "layer": int(layer), "metric": "cosine similarity", "value": value})
    if not metric_rows:
        raise ValueError(f"no scale sweep metrics found in {directory}")
    return pd.DataFrame(metric_rows), response_rows


def metric_figure(frame: pd.DataFrame, target_title: str) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=("delta output norm", "Cosine similarity", target_title, "Steering effect", "Factual match", ""),
    )
    scales = sorted(frame.scale.unique())
    colors = {scale: color for scale, color in zip(scales, px.colors.sample_colorscale("Viridis", [i / max(len(scales) - 1, 1) for i in range(len(scales))]))}

    for metric in ("delta output norm", "cosine similarity"):
        data = frame[frame.metric == metric]
        if metric == "delta output norm":
            data = data.groupby(["scale", "prefix_length"], as_index=False).value.mean()
        else:
            data = data.groupby(["scale", "prefix_length"], as_index=False).value.mean()
        for scale in scales:
            subset = data[data.scale == scale].sort_values("prefix_length")
            if subset.empty:
                continue
            x_values = subset.prefix_length.to_numpy(dtype=float) + 1.0
            row, col = (1, 1) if metric == "delta output norm" else (1, 2)
            figure.add_scatter(
                x=x_values,
                y=subset.value,
                mode="lines+markers",
                name=f"scale {scale:g}",
                legendgroup=f"scale-{scale}",
                showlegend=False,
                line=dict(color=colors[scale]),
                marker=dict(color=colors[scale]),
                hovertemplate="scale=%{legendgroup}<br>prefix=%{customdata} tokens<br>value=%{y}<extra></extra>",
                customdata=subset.prefix_length,
                row=row,
                col=col,
            )

    for metric, dash in (("baseline share", "dash"), ("target share", "solid")):
        shares = frame[frame.metric == metric]
        for scale in scales:
            subset = shares[shares.scale == scale].sort_values("prefix_length")
            if subset.empty:
                continue
            figure.add_scatter(
                x=subset.prefix_length.to_numpy(dtype=float) + 1.0,
                y=subset.value,
                mode="lines+markers",
                name=f"{metric}, scale {scale:g}",
                legendgroup=f"share-{scale}",
                showlegend=False,
                line=dict(color=colors[scale], dash=dash),
                marker=dict(color=colors[scale]),
                hovertemplate=f"{metric}<br>scale=%{{legendgroup}}<br>prefix=%{{customdata}} tokens<br>share=%{{y}}<extra></extra>",
                customdata=subset.prefix_length,
                row=2,
                col=1,
            )

    effects = frame[frame.metric == "steering effect"]
    for scale in scales:
        subset = effects[effects.scale == scale].sort_values("prefix_length")
        if subset.empty:
            continue
        figure.add_scatter(
            x=subset.prefix_length.to_numpy(dtype=float) + 1.0,
            y=subset.value,
            mode="lines+markers",
            name=f"scale {scale:g}",
            legendgroup=f"effect-{scale}",
            showlegend=False,
            line=dict(color=colors[scale]),
            marker=dict(color=colors[scale]),
            hovertemplate="scale=%{legendgroup}<br>prefix=%{customdata} tokens<br>effect=%{y}<extra></extra>",
            customdata=subset.prefix_length,
            row=2,
            col=2,
        )

    factual_matches = frame[frame.metric == "factual match"]
    for scale in scales:
        subset = factual_matches[factual_matches.scale == scale].sort_values("prefix_length")
        if subset.empty:
            continue
        figure.add_scatter(
            x=subset.prefix_length.to_numpy(dtype=float) + 1.0,
            y=subset.value,
            mode="lines+markers",
            name=f"scale {scale:g}",
            legendgroup=f"factual-match-{scale}",
            showlegend=False,
            line=dict(color=colors[scale]),
            marker=dict(color=colors[scale]),
            hovertemplate="scale=%{legendgroup}<br>prefix=%{customdata} tokens<br>factual match=%{y}<extra></extra>",
            customdata=subset.prefix_length,
            row=3,
            col=1,
        )

    for row, column in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1)):
        figure.update_xaxes(title="filler prefix length (tokens)", type="log", row=row, col=column)
    figure.update_yaxes(title="delta norm", row=1, col=1)
    figure.update_yaxes(title="cosine", row=1, col=2)
    figure.update_yaxes(title="share", range=[0, 1], row=2, col=1)
    figure.update_yaxes(title="steering effect", range=[-1, 1], row=2, col=2)
    figure.update_yaxes(title="factual match", range=[0, 1], row=3, col=1)

    for scale in scales:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"scale {scale:g}",
                legendgroup=f"legend-scale-{scale}",
                showlegend=True,
                line=dict(color=colors[scale]),
                marker=dict(color=colors[scale]),
                visible="legendonly",
            )
        )

    figure.update_layout(height=1200, width=1500, title="Forgetting scale sweep", legend_title="scale", template="plotly_white")
    return figure


def response_figure(rows: list[dict[str, Any]]) -> str:
    grouped: dict[float, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["scale"], {}).setdefault(row["prefix_length"], []).append(row)

    sections: list[str] = []
    for scale in sorted(grouped):
        scale_sections: list[str] = [f"<details open><summary><b>scale={scale:g}</b></summary>"]
        for prefix_length in sorted(grouped[scale]):
            values = sorted(grouped[scale][prefix_length], key=lambda item: (item["condition"] != "baseline", item["condition"]))
            scale_sections.append(f"<details><summary>prefix={prefix_length} tokens</summary>")
            if "concept_score" in values[0]:
                scale_sections.append("<table><tr><th>condition</th><th>score</th><th>question</th><th>response</th></tr>")
                for row in values:
                    scale_sections.append(
                        f"<tr><td>{html.escape(row['condition'])}</td><td>{row['concept_score']}</td><td>{html.escape(row['question'])}</td><td>{html.escape(row['response'])}</td></tr>"
                    )
            else:
                scale_sections.append("<table><tr><th>condition</th><th>detected</th><th>confidence</th><th>factual match</th><th>response</th></tr>")
                for row in values:
                    scale_sections.append(
                        f"<tr><td>{html.escape(row['condition'])}</td><td>{html.escape(row['detected_language'])}</td><td>{row['language_confidence']:.3f}</td><td>{row.get('factual_match', '')}</td><td>{html.escape(row['response'])}</td></tr>"
                    )
            scale_sections.append("</table></details>")
        scale_sections.append("</details>")
        sections.append("\n".join(scale_sections))
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics, responses = load_metrics(args.input)
    response_frame = pd.DataFrame(responses)
    if "concept_score" in response_frame:
        response_frame["is_target"] = response_frame.concept_score
        target = str(response_frame.target_concept.iloc[0])
        target_title = f"Baseline and steered: {target}"
    else:
        response_frame["is_target"] = response_frame.detected_language.eq("ru")
        target_title = "Baseline and steered target language"
    shares = (
        response_frame
        .groupby(["scale", "prefix_length", "condition"])
        .is_target.mean()
        .unstack("condition")
        .reset_index()
    )
    shares["baseline share"] = shares.get("baseline", 0.0)
    shares["target share"] = shares.get("steered", 0.0)
    shares["steering effect"] = shares.get("steered", 0.0) - shares.get("baseline", 0.0)
    extra = shares[["scale", "prefix_length", "baseline share", "target share", "steering effect"]].melt(id_vars=["scale", "prefix_length"], var_name="metric", value_name="value")
    factual_matches = pd.DataFrame(columns=["scale", "prefix_length", "value", "metric"])
    if "factual_match" in response_frame:
        factual_matches = (
            response_frame.query("condition == 'steered'")
            .groupby(["scale", "prefix_length"], as_index=False)
            .factual_match.mean()
            .rename(columns={"factual_match": "value"})
            .assign(metric="factual match")
        )
    figure = metric_figure(pd.concat([metrics, extra, factual_matches], ignore_index=True), target_title)
    document = "<html><head><meta charset='utf-8'><title>Forgetting scale sweep</title><style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse;margin-bottom:2rem;width:100%}th,td{border:1px solid #ccc;padding:.35rem;text-align:left;vertical-align:top}td:last-child{max-width:70rem;white-space:pre-wrap}</style></head><body><h1>Forgetting scale sweep</h1>"
    document += figure.to_html(full_html=False, include_plotlyjs="cdn")
    document += "<h2>Model responses</h2>" + response_figure(responses) + "</body></html>"
    args.output.write_text(document)


if __name__ == "__main__":
    main()
