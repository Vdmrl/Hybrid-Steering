"""Score ablation JSONL files and write a compact HTML comparison."""

import argparse
import json
import re
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

from gdn_interp import LanguageDetector


def repeated_fraction(text: str) -> float:
    """Return the share of duplicate word four-grams."""
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read generated responses from one JSONL result file."""
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def score(path: Path, detector: LanguageDetector) -> dict[str, Any]:
    """Summarize one generated-response file."""
    rows = read_rows(path)
    labels = []
    repeats = []
    for row in rows:
        label = row.get("detected_language")
        if not isinstance(label, str):
            label = detector.label(row["response"])
        labels.append(label)
        repeats.append(repeated_fraction(row["response"]))
    first = rows[0]
    counts = Counter(labels)
    return {
        "path": str(path.relative_to(path.parents[2])),
        "n": len(rows),
        "target": first["target"],
        "target_count": counts[first["target"]],
        "counts": dict(counts),
        "repetition_flags": sum(value > 0.2 for value in repeats),
        "mean_repeat": sum(repeats) / len(repeats),
        "truncated": sum(row["truncated"] for row in rows),
        "mean_relative_intervention": sum(row.get("state", {}).get("mean_relative_intervention", 0.0) for row in rows) / len(rows),
        "rank": first["rank"],
        "group": first["group"],
        "period": first["period"],
        "scale": first["scale"],
        "position": first.get("position", "pre-final"),
        "mode": first.get("mode", "add"),
    }


def full_prompt(row: dict[str, Any]) -> str:
    """Return the context and question shown to the model."""
    return f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}"


def generated_texts(paths: list[Path]) -> str:
    """Render all generated responses grouped by sweep condition."""
    sections = []
    for path in paths:
        rows = read_rows(path)
        table = "".join(
            "<tr>"
            f"<td>{index}</td><td><pre>{escape(full_prompt(row))}</pre></td><td>{escape(str(row['response']))}</td>"
            f"<td>{escape(str(row.get('detected_language', 'unknown')))}</td>"
            "</tr>"
            for index, row in enumerate(rows, start=1)
        )
        sections.append(
            f"<details><summary>{escape(str(path.relative_to(path.parents[2])))} ({len(rows)} responses)</summary>"
            "<table><tr><th>#</th><th>Full prompt</th><th>Generated response</th><th>Language</th></tr>"
            f"{table}</table></details>"
        )
    return "<h2>Generated responses</h2>" + "".join(sections)


def charts(summaries: list[dict[str, Any]]) -> str:
    """Render target-language detection curves grouped by position and rank."""
    groups = {}
    for row in summaries:
        if row["mode"] == "baseline":
            continue
        groups.setdefault((row["position"], row["rank"]), []).append(row)
    rendered = []
    width, height = 720, 320
    left, top, right, bottom = 64, 24, 24, 48
    plot_width, plot_height = width - left - right, height - top - bottom
    for (position, rank), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["scale"])
        x_values = [row["scale"] for row in rows]
        y_values = [row["target_count"] / row["n"] for row in rows]
        minimum, maximum = min(x_values), max(x_values)
        x_span = maximum - minimum or 1.0
        points = []
        for scale, share in zip(x_values, y_values):
            x = left + (scale - minimum) / x_span * plot_width
            y = top + (1.0 - share) * plot_height
            points.append((x, y))
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3"/>' for x, y in points)
        x_labels = "".join(
            f'<text x="{x:.1f}" y="{height - 16}" text-anchor="middle">{scale:g}</text>'
            for scale, (x, _) in zip(x_values, points)
        )
        y_labels = "".join(
            f'<text x="{left - 10}" y="{top + (1 - share) * plot_height + 4}" text-anchor="end">{share:.0%}</text>'
            for share in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
        grid = "".join(
            f'<line x1="{left}" y1="{top + (1 - share) * plot_height:.1f}" x2="{width - right}" y2="{top + (1 - share) * plot_height:.1f}"/>'
            for share in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
        label = "full" if rank == 0 else str(rank)
        rendered.append(
            f"<figure><figcaption>Position {escape(str(position))}, rank {label}</figcaption>"
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Target language detection by scale">'
            f'<g class="grid">{grid}</g><g class="labels">{y_labels}{x_labels}</g>'
            f'<polyline points="{polyline}"/><g class="dots">{dots}</g></svg></figure>'
        )
    return "<h2>Target-language detection by scale</h2><div class=charts>" + "".join(rendered) + "</div>"


def main() -> None:
    """Parse result files, score language and repetition, and render HTML."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detector = LanguageDetector("en")
    paths = sorted(path for path in args.root.rglob("*.jsonl") if path.stat().st_size)
    summaries = [score(path, detector) for path in paths]
    summaries.sort(key=lambda row: (row["position"], row["mode"], row["rank"], row["group"], row["scale"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    table = "".join(
        "<tr>"
        f"<td>{escape(row['path'])}</td><td>{row['rank']}</td><td>{escape(row['group'])}</td><td>{escape(row['position'])}</td><td>{escape(row['mode'])}</td><td>{row['scale']:g}</td>"
        f"<td>{row['target_count']}/{row['n']} ({row['target_count'] / row['n']:.1%})</td><td>{escape(str(row['counts']))}</td>"
        f"<td>{row['repetition_flags']}/{row['n']} ({row['mean_repeat']:.1%} mean)</td><td>{row['truncated']}/{row['n']}</td><td>{row['mean_relative_intervention']:.3g}</td>"
        "</tr>"
        for row in summaries
    )
    responses = generated_texts(paths)
    plot = charts(summaries)
    args.output.write_text(
        "<!doctype html><meta charset=utf-8><title>GDN steering ablations</title>"
        "<style>body{font:15px system-ui;max-width:1600px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap;min-width:420px;margin:0}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}figure{margin:0;border:1px solid #ddd;padding:12px}figcaption{font-weight:600;margin-bottom:8px}svg{width:100%;height:auto}svg polyline{fill:none;stroke:#1769aa;stroke-width:3}svg circle{fill:#1769aa}.grid line{stroke:#d8dee4;stroke-width:1}.labels{fill:#4b5563;font-size:12px}</style>"
        "<h1>Context-bearing SQuAD steering ablations</h1>"
        "<p>Each prompt supplies a truncated SQuAD passage and its question. Language is Lingua's dominant label among ten supported languages. A repetition flag means more than 20% duplicate word four-grams. Relative intervention is the mean Frobenius norm of the applied change divided by the current state norm; initial-state interventions report zero because no post-prefill change is applied.</p>"
        "<h2>Full ablation table</h2>"
        "<table><tr><th>Run</th><th>Rank</th><th>Layers</th><th>Position</th><th>Mode</th><th>Scale</th><th>Target language</th><th>Labels</th><th>Repetition</th><th>No EOS</th><th>Relative change</th></tr>"
        + table
        + "</table>"
        + plot
        + responses
    )


if __name__ == "__main__":
    main()
