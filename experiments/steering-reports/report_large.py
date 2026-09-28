"""Re-score saved large sweeps and write an offline HTML report.

uv run --with lingua-language-detector python experiments/steering/report_large.py
"""
import json
import re
from collections import Counter
from html import escape
from pathlib import Path

from gdn_interp import LanguageDetector


def main():
    root = Path(__file__).resolve().parents[2] / 'artifacts/steering/large'
    summaries, sections = [], []
    reference = None
    for target in ('ru', 'fr'):
        detector = LanguageDetector(target)
        for scale in (0,1,2,4):
            p = root / target / f'r0_all_p0_s{scale}.jsonl'
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            assert len(rows) == 200
            questions = [r['question'] for r in rows]
            if reference is None: reference = questions
            assert questions == reference, 'Question sets differ'
            for row in rows:
                row['language_lingua'] = detector.label(row['response'])
                row['language_confidence'] = detector.confidence(row['response'])
                words = re.findall(r'\w+',row['response'].lower())
                grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
                row['repeated_4gram_fraction'] = 1-len(set(grams))/len(grams) if grams else 0
            counts = Counter(r['language_lingua'] for r in rows)
            summary = dict(target=target,scale=scale,n=len(rows),detected=dict(counts),target_count=counts[target],truncated=sum(r['truncated'] for r in rows),repetition_flags=sum(r['repeated_4gram_fraction']>.2 for r in rows),mean_relative_intervention=rows[0]['state']['mean_relative_intervention'])
            summaries.append(summary)
            (root/target/f'scored_s{scale}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
            print(json.dumps(summary),flush=True)
            examples = ''.join(f'<tr><td>{escape(r["question"])}</td><td>{escape(r["response"])}</td><td>{r["language_lingua"]} ({r["language_confidence"]:.2f})</td><td>{r["truncated"]}</td><td>{r["repeated_4gram_fraction"]:.0%}</td></tr>' for r in rows)
            sections.append(f'<details><summary>{target}, scale {scale}: {counts[target]}/200 target-language detections</summary><table><tr><th>Question</th><th>Response</th><th>Lingua</th><th>Token cap</th><th>Repeated 4-grams</th></tr>{examples}</table></details>')
    table = ''.join(f'<tr><td>{s["target"]}</td><td>{s["scale"]}</td><td>{s["target_count"]}/200 ({s["target_count"]/2:.1f}%)</td><td>{s["truncated"]}</td><td>{s["repetition_flags"]}</td><td>{s["mean_relative_intervention"]:.3g}</td></tr>' for s in summaries)
    (root/'summary.json').write_text(json.dumps(summaries,indent=2))
    (root/'report.html').write_text('''<!doctype html><meta charset="utf-8"><title>Large GDN steering evaluation</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ddd;text-align:left;vertical-align:top}td:nth-child(2){white-space:pre-wrap}details{margin:22px 0}summary{cursor:pointer}</style><h1>English→Russian and English→French: 200 questions each</h1>
<p>Same 200 unique SQuAD validation questions (seed 20260911, ≤25 words) in every condition, without passages. Eight matched bilingual answers calibrate full empirical deltas; intervention is applied once across all GDN layers before the final prompt token. Greedy decoding, 128-token cap, batch size 200. RU: GPU 0 on airi.gpu; FR: GPU 2 on airi.gpu3. The calibration English answers and topics are shared; target translations differ.</p>
<p>Lingua selects among EN, FR, RU, DE, ES, IT, PT, NL, UK, PL from the complete response. Scores are detector confidence, not calibrated correctness probabilities. Dominant-language detection can hide mixed text or misclassify short names. The raw generator's Cyrillic metric is unsuitable for French. Token-cap flags mean no EOS within 128 tokens. Repetition flags mean &gt;20% duplicate word 4-grams; this is a heuristic, not a semantic-quality rating.</p>
<p>SQuAD questions often depend on omitted context, so these runs test language transfer, not factual answer accuracy. Do not compare their rates directly with the earlier 20 everyday questions as if only sample size changed. Model outputs below are unedited. Scale selection on this set is exploratory. The H200 and H100 hosts produced different text for some unsteered questions; compare each target to its own baseline, not as a strictly controlled cross-language hardware comparison. Inspection found prompt copying and unsupported claims in some steered answers, especially at stronger scales.</p><table><tr><th>Target</th><th>Scale</th><th>Target language</th><th>Token cap</th><th>Repetition flags</th><th>Mean relative intervention</th></tr>'''+table+'</table>'+''.join(sections))


if __name__ == '__main__': main()
