"""Create one portable HTML report from judged concept-steering responses."""
import argparse, html, json
from collections import defaultdict
from pathlib import Path

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", type=Path, required=True); args = p.parse_args()
    files = list(args.run_dir.glob("*/judgments.jsonl")) or [args.run_dir / "judgments.jsonl"]
    rows = [json.loads(line) for file in files for line in file.read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows: grouped[(row["concept"], row["scale"])].append(row["score"])
    bars = []
    for (concept, scale), scores in sorted(grouped.items()):
        mean = sum(scores) / len(scores); width = abs(mean) * 48
        bars.append(f"<tr><td>{html.escape(concept)}</td><td>{scale:+g}</td><td>{mean:+.2f}</td><td><div class='bar {'pos' if mean >= 0 else 'neg'}' style='width:{width:.1f}%'></div></td></tr>")
    examples = "".join(f"<tr><td>{html.escape(r['concept'])}</td><td>{r['example_id']}</td><td>{r['scale']:+g}</td><td>{r.get('mode') or 'baseline'}</td><td>{r['score']:+d}</td><td>{html.escape(r['prompt'])}</td><td>{html.escape(r['response'])}</td><td>{html.escape(r['reason'])}</td></tr>" for r in rows)
    page = f"""<!doctype html><meta charset=utf-8><title>Concept steering</title><style>body{{font:14px system-ui;margin:2rem;color:#18212b}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}}td{{white-space:pre-wrap}}.bar{{height:1rem;border-radius:3px}}.pos{{background:#277da1}}.neg{{background:#e76f51}}</style><h1>Concept steering benchmark</h1><p>Mean judge score: +1 is the first concept, −1 the second; scale 0 is baseline. See each row's mode for whether steering was one-shot or repeated.</p><h2>Detection</h2><table><tr><th>Concept</th><th>Scale</th><th>Mean</th><th>Magnitude</th></tr>{''.join(bars)}</table><h2>All answers</h2><table><tr><th>Concept</th><th>ID</th><th>Scale</th><th>Mode</th><th>Score</th><th>Prompt</th><th>Response</th><th>Judge rationale</th></tr>{examples}</table>"""
    (args.run_dir / "report.html").write_text(page)

if __name__ == "__main__": main()
