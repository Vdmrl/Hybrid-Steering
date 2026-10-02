"""Render language, repetition, and sample comparisons from steering JSONL files."""

import argparse
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gdn_interp import LanguageDetector


@dataclass
class Result:
    """One scored steering condition."""

    path: str
    rows: list[dict[str, Any]]
    labels: list[str]
    repeats: list[float]

    @property
    def first(self) -> dict[str, Any]:
        return self.rows[0]

    @property
    def target_rate(self) -> float:
        return sum(label == self.first["target"] for label in self.labels) / len(self.rows)


def repeat_fraction(text: str) -> float:
    """Return the fraction of duplicate word four-grams."""
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def read_result(path: Path, root: Path, detector: LanguageDetector) -> Result:
    """Score one JSONL response file."""
    rows = [json.loads(line) for line in path.read_text().split("\n") if line]
    labels = []
    for row in rows:
        labels.append(detector.label(row["response"]))
    return Result(str(path.relative_to(root)), rows, labels, [repeat_fraction(row["response"]) for row in rows])


def label(result: Result) -> str:
    """Return a compact heatmap row label."""
    row = result.first
    regime = result.path.split("/", 1)[0]
    return f"{regime} · {row.get('mode', 'add')} · {row.get('position', 'pre-final')} · r{row['rank']} · {row['group']}"


def matrix(results: list[Result], labels: list[str], axis: str) -> tuple[list[str], list[list[float | None]], list[list[float | None]]]:
    """Return shared-scale language and repetition matrices for one comparison."""
    scales = sorted({result.first["scale"] for result in results})
    language = [[None for _ in scales] for _ in labels]
    repeat = [[None for _ in scales] for _ in labels]
    for result in results:
        row = labels.index(str(result.first[axis]))
        column = scales.index(result.first["scale"])
        language[row][column] = result.target_rate
        repeat[row][column] = sum(result.repeats) / len(result.repeats)
    return [f"{scale:g}" for scale in scales], language, repeat


def axis_heatmap(results: list[Result], regime: str, title: str, axis: str, filters: dict[str, Any]) -> str:
    """Render language and repetition for one varying experimental axis."""
    chosen = [result for result in results if result.path.split("/", 1)[0] == regime and all((result.path.split("/")[1] in value if key == "branch" else result.first.get(key) == value) for key, value in filters.items())]
    labels = sorted({str(result.first[axis]) for result in chosen}, key=lambda value: float(value) if value.replace(".", "", 1).isdigit() else value)
    scales, language, repeat = matrix(chosen, labels, axis)
    target = chosen[0].first["target"]
    language_name = {"ru": "Russian", "fr": "French"}[target]
    figure = make_subplots(rows=1, cols=2, subplot_titles=(f"Dominant {language_name}", "Repeated four-grams"), shared_yaxes=True)
    figure.add_trace(go.Heatmap(z=language, x=scales, y=labels, colorscale="Viridis", zmin=0, zmax=1, colorbar={"title": target.upper(), "x": 0.45}, hovertemplate=f"{axis}=%{{y}}<br>scale=%{{x}}<br>{language_name}=%{{z:.1%}}<extra></extra>"), row=1, col=1)
    figure.add_trace(go.Heatmap(z=repeat, x=scales, y=labels, colorscale="Viridis", zmin=0, zmax=1, colorbar={"title": "Repeat", "x": 1.02}, hovertemplate=f"{axis}=%{{y}}<br>scale=%{{x}}<br>repeat=%{{z:.1%}}<extra></extra>"), row=1, col=2)
    figure.update_layout(title=f"{regime}: {title}", height=max(280, 70 * len(labels)), margin={"l": 100, "r": 100, "t": 60, "b": 45})
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


def decomposed_heatmaps(results: list[Result]) -> str:
    """Render one-factor heatmaps for every prompt regime."""
    comparisons = [
        ("Rank", "rank", {"branch": {"baseline", "rank2", "rank4", "full_rank"}, "mode": "add", "position": "pre-final", "group": "all"}),
        ("Layer group", "group", {"branch": {"baseline", "layer_ranges"}, "mode": "add", "position": "pre-final", "rank": 1}),
        ("Injection position", "position", {"branch": {"baseline", "post_final", "initial", "after_chunk"}, "mode": "add", "group": "all", "rank": 1}),
        ("Intervention mode", "mode", {"branch": {"baseline", "clamp"}, "position": "pre-final", "group": "all", "rank": 1}),
    ]
    sections = []
    for regime in sorted({result.path.split("/", 1)[0] for result in results}):
        sections.append(f"<h2>{escape(regime)}</h2>")
        for title, axis, filters in comparisons:
            sections.append(axis_heatmap(results, regime, title, axis, filters))
        calibration = [result for result in results if result.path.split("/", 1)[0] == regime and result.first["mode"] == "add" and result.first["position"] == "pre-final" and result.first["group"] == "all" and result.first["rank"] == 1 and result.path.split("/")[1] in {"baseline", "opus_calibration"}]
        labels = ["matched-8" if "/baseline/" in result.path else "OPUS-200" for result in calibration]
        for result, name in zip(calibration, labels, strict=True):
            result.rows[0]["calibration"] = name
        sections.append(axis_heatmap(calibration, regime, "Calibration source", "calibration", {}))
    return "".join(sections)


def summary_table(results: list[Result]) -> str:
    """Render all conditions as an HTML table."""
    rows = []
    for result in sorted(results, key=lambda item: (item.path, item.first["scale"])):
        row = result.first
        repeats = sum(value > 0.2 for value in result.repeats)
        rows.append(f"<tr><td>{escape(result.path)}</td><td>{len(result.rows)}</td><td>{row['rank']}</td><td>{escape(row['group'])}</td><td>{escape(row.get('position', 'pre-final'))}</td><td>{escape(row.get('mode', 'add'))}</td><td>{row['scale']:g}</td><td>{result.target_rate:.1%}</td><td>{repeats}/{len(result.rows)}</td><td>{sum(row['truncated'] for row in result.rows)}/{len(result.rows)}</td></tr>")
    language_name = {"ru": "Russian", "fr": "French"}[results[0].first["target"]]
    return f"<table><tr><th>Run</th><th>N</th><th>Rank</th><th>Layer group</th><th>Position</th><th>Mode</th><th>Scale</th><th>{language_name}</th><th>Repeat flags</th><th>No EOS</th></tr>" + "".join(rows) + "</table>"


def samples(results: list[Result]) -> str:
    """Compare the strongest condition for every prompt regime and layer group."""
    keys = sorted({(result.path.split("/", 1)[0], result.first["group"]) for result in results})
    best = {
        key: max((result for result in results if (result.path.split("/", 1)[0], result.first["group"]) == key), key=lambda result: result.target_rate)
        for key in keys
    }
    target = results[0].first["target"]
    headings = "".join(f"<th>{escape(regime)} / {escape(group)}: {escape(label(best[(regime, group)]))}, scale {best[(regime, group)].first['scale']:g}, {best[(regime, group)].target_rate:.0%} {target.upper()}</th>" for regime, group in keys)
    body = []
    for index in range(min(12, len(next(iter(best.values())).rows))):
        question = next(iter(best.values())).rows[index]["question"]
        cells = "".join(f"<td><small>{escape(best[(regime, group)].labels[index])}</small><br>{escape(best[(regime, group)].rows[index]['response'])}</td>" for regime, group in keys)
        body.append(f"<tr><th>{escape(question)}</th>{cells}</tr>")
    language_name = {"ru": "Russian", "fr": "French"}[target]
    return f"<h2>Matched prompt samples</h2><p>Each column is the best {language_name}-detection condition within a prompt regime and layer group. Rows share the same question.</p><table><tr><th>Question</th>" + headings + "</tr>" + "".join(body) + "</table>"


def main() -> None:
    """Read ablation artifacts and write the visual report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detector = LanguageDetector("en")
    results = [read_result(path, args.root, detector) for path in sorted(args.root.rglob("*.jsonl")) if path.stat().st_size]
    if not results:
        parser.error(f"no completed JSONL files found under {args.root}")
    targets = {result.first["target"] for result in results}
    if len(targets) != 1:
        parser.error(f"mixed target languages under {args.root}: {sorted(targets)}")
    language_name = {"ru": "Russian", "fr": "French"}[targets.pop()]
    metric_note = " The generator's Cyrillic fields are ignored for French." if language_name == "French" else ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("<!doctype html><meta charset=utf-8><title>GDN steering ablations</title><style>body{font:15px system-ui;max-width:1900px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border:1px solid #ddd;text-align:left;vertical-align:top}td{white-space:pre-wrap}small{color:#666}</style><h1>GDN language-steering ablations</h1>" + f"<p>{language_name} rate is Lingua's dominant-language label.{metric_note} Each heatmap varies one factor vertically while holding the stated baseline settings fixed. A repeat flag means more than 20% duplicate word four-grams. No EOS means generation reached its token cap. These metrics do not assess factual correctness. Blank cells were not run.</p>" + decomposed_heatmaps(results) + "<h2>All completed conditions</h2>" + summary_table(results) + samples(results))


if __name__ == "__main__":
    main()
