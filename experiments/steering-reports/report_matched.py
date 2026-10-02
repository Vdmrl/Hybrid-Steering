"""Write an offline Lingua-scored report for a SQuAD steering sweep."""

import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

from gdn_interp import LanguageDetector

def detect(rows: list[dict[str, Any]]) -> None:
    detector = LanguageDetector(str(rows[0]["target"]))
    for row in rows:
        row["detected_language"] = detector.label(row["response"])
        row["language_confidence"] = detector.confidence(row["response"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an offline SQuAD steering report.")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summaries, sections = [], []
    for path in sorted(args.directory.glob("scale_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            continue
        detect(rows)
        counts = Counter(row["detected_language"] for row in rows)
        scale = rows[0]["scale"]
        summaries.append(f"<tr><td>{scale:g}</td><td>{len(rows)}</td><td>{escape(json.dumps(dict(counts)))}</td></tr>")
        examples = "".join(
            f'<tr><td>{escape(row["question"])}</td><td>{escape(row["response"])}</td><td>{row["detected_language"]} ({row["language_confidence"]:.0%})</td></tr>'
            for row in rows
        )
        sections.append(
            f'<details><summary>scale {scale:g}: {escape(json.dumps(dict(counts)))}</summary><table><tr><th>Question</th><th>Response</th><th>Lingua</th></tr>{examples}</table></details>'
        )
        path.with_suffix(".scored.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    html = """<!doctype html><meta charset=\"utf-8\"><title>SQuAD GDN steering</title>
<style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:left;vertical-align:top}td:nth-child(2){white-space:pre-wrap}details{margin:18px 0}</style>
<h1>SQuAD GDN steering</h1><p>Lingua selects among ten languages from each complete response. Confidence is detector confidence, not answer quality.</p>
<table><tr><th>Scale</th><th>Responses</th><th>Detected languages</th></tr>"""
    (args.directory / "report.html").write_text(html + "".join(summaries) + "</table>" + "".join(sections))
    print(f"Wrote {args.directory / 'report.html'}")


if __name__ == "__main__":
    main()
