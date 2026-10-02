"""Render experimental clamping JSONL files as a compact HTML report."""
import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("directory", type=Path); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    sections = []
    for path in sorted(a.directory.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        counts = Counter(row["detected_language"] for row in rows)
        state = rows[0].get("state", {}) if rows else {}
        stats = "<br>".join(f"{escape(k)}: {v:.3g}" for k, v in list(state.items())[:8])
        examples = "".join(f"<tr><td>{escape(r['question'])}</td><td>{escape(r['response'])}</td><td>{r['detected_language']}</td></tr>" for r in rows)
        sections.append(f"<section><h2>{escape(path.stem)}</h2><p>{len(rows)} examples; detected language: {escape(str(dict(counts)))}</p><p>Mean state coordinates after the intervention:<br>{stats}</p><table><tr><th>Question</th><th>Generation</th><th>Detected</th></tr>{examples}</table></section>")
    a.output.write_text("<!doctype html><meta charset=utf-8><style>body{font:15px system-ui;max-width:1400px;margin:auto}section{margin:24px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:left;vertical-align:top}</style><h1>GDN language-clamping experiments</h1><p>Initial screen: EN prompts toward the natural RU coordinate; detection is Cyrillic-letter share ≥20%.</p>" + "".join(sections))


if __name__ == "__main__": main()
