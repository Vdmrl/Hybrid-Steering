"""Render norm-aware steering results and natural-state singular statistics."""

import argparse
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from gdn_interp import LanguageDetector


@dataclass
class Result:
    """One method and prompt-regime scale sweep."""

    regime: str
    method: str
    rows: list[dict[str, Any]]
    labels: list[str]
    repeats: list[float]

    def rate(self) -> float:
        return sum(label == "ru" for label in self.labels) / len(self.rows)


def repeat_fraction(text: str) -> float:
    """Return the fraction of repeated word four-grams."""
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def read(path: Path, detector: LanguageDetector) -> Result:
    """Read and score one JSONL condition file."""
    rows = [json.loads(line) for line in path.read_text().split("\n") if line]
    labels = []
    for row in rows:
        labels.append(detector.label(row["response"]))
    regime, method = path.parts[-3:-1]
    return Result(regime, method, rows, labels, [repeat_fraction(row["response"]) for row in rows])


def sweep_heatmap(results: list[Result], regime: str) -> str:
    """Plot language and repetition over scale for one prompt regime."""
    chosen = [result for result in results if result.regime == regime]
    methods = sorted({result.method for result in chosen})
    scales = sorted({result.rows[0]["scale"] for result in chosen})
    language = [[None for _ in scales] for _ in methods]
    repeat = [[None for _ in scales] for _ in methods]
    for result in chosen:
        row, column = methods.index(result.method), scales.index(result.rows[0]["scale"])
        language[row][column] = result.rate()
        repeat[row][column] = sum(result.repeats) / len(result.repeats)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Dominant Russian", "Repeated four-grams"), shared_yaxes=True)
    figure.add_trace(go.Heatmap(z=language, x=[f"{scale:g}" for scale in scales], y=methods, colorscale="Viridis", zmin=0, zmax=1, colorbar={"title": "RU", "x": 0.45}), row=1, col=1)
    figure.add_trace(go.Heatmap(z=repeat, x=[f"{scale:g}" for scale in scales], y=methods, colorscale="Magma", zmin=0, zmax=1, colorbar={"title": "Repeat", "x": 1.02}), row=1, col=2)
    figure.update_layout(title=regime, height=360, margin={"l": 130, "r": 100, "t": 60, "b": 45})
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


def reference_plots(path: Path, regime: str) -> str:
    """Plot per-head average natural-state scale statistics."""
    reference = torch.load(path, map_location="cpu", weights_only=True)
    layers = sorted(reference)
    singular = [reference[layer]["singular"][:, 0].tolist() for layer in layers]
    frobenius = [reference[layer]["frobenius"].tolist() for layer in layers]
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Mean leading singular value of S", "Mean Frobenius norm of S"))
    figure.add_trace(go.Heatmap(z=singular, x=list(range(len(singular[0]))), y=layers, colorscale="Viridis"), row=1, col=1)
    figure.add_trace(go.Heatmap(z=frobenius, x=list(range(len(frobenius[0]))), y=layers, colorscale="Viridis"), row=1, col=2)
    figure.update_layout(height=750, margin={"l": 80, "r": 50, "t": 60, "b": 45})
    figure.update_layout(title=regime.replace("_", " "))
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


def examples(results: list[Result], regime: str) -> str:
    """Show matched outputs from each method at its strongest Russian scale."""
    best = {method: max((result for result in results if result.regime == regime and result.method == method), key=lambda result: result.rate()) for method in {result.method for result in results if result.regime == regime}}
    methods = sorted(best)
    headings = "".join(f"<th>{escape(method)}<br>scale {best[method].rows[0]['scale']:g}, {best[method].rate():.0%} RU</th>" for method in methods)
    rows = []
    for index in range(10):
        question = best[methods[0]].rows[index]["question"]
        cells = "".join(f"<td><small>{escape(best[method].labels[index])}</small><br>{escape(best[method].rows[index]['response'])}</td>" for method in methods)
        rows.append(f"<tr><th>{escape(question)}</th>{cells}</tr>")
    return f"<h2>{escape(regime)} examples</h2><table><tr><th>Question</th>{headings}</tr>{''.join(rows)}</table>"


def main() -> None:
    """Write the normalization report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detector = LanguageDetector("ru")
    results = [read(path, detector) for path in args.root.rglob("*.jsonl")]
    references = {
        path.parent.parent.name: path
        for path in args.root.rglob("state_reference.pt")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_statistics = "".join(
        f"<h2>Natural GDN state statistics: {escape(regime.replace('_', ' '))}</h2>"
        + reference_plots(path, regime)
        for regime, path in sorted(references.items())
    )
    args.output.write_text("<!doctype html><meta charset=utf-8><title>GDN normalization study</title><style>body{font:15px system-ui;max-width:1900px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%;margin:22px 0}td,th{padding:8px;border:1px solid #ddd;text-align:left;vertical-align:top}td{white-space:pre-wrap}small{color:#666}</style><h1>GDN steering scale normalization</h1><p>Raw uses the empirical delta directly. Frobenius and spectral normalization set each head’s delta scale from the average natural pre-steering state on these 50 prompts. Norm-preserving steering rescales each post-intervention head to its original Frobenius norm. Russian is Lingua’s dominant label.</p>" + state_statistics + "".join(sweep_heatmap(results, regime) + examples(results, regime) for regime in sorted({result.regime for result in results})))


if __name__ == "__main__":
    main()
