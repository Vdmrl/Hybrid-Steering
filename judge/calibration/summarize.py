import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
references=json.loads((ROOT/'calibration/cases-private.json').read_text(encoding='utf-8'))
rubric=json.loads((ROOT/'ready_judge/resources/concepts/features.yaml').read_text(encoding='utf-8'))
out={'reference_author':'one Codex agent; not independent human consensus','status':'exploratory pilot', 'features':{}}
for name in ('numbered','french','complexity','fairy_tale'):
    ref=[r for r in references if r['feature']==name]
    path=ROOT/'runs/calibration'/name/'scores.jsonl'
    rows=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    scores={r['answer_id']:r for r in rows}
    pairs=[(r,scores[r['answer_id']]) for r in ref if r['answer_id'] in scores]
    summary={'n_expected':len(ref),'n_scored':len(pairs)}
    for difficulty in ('obvious','hard'):
        items=[(r,s) for r,s in pairs if r['difficulty']==difficulty]
        summary[difficulty]={'n':len(items),'exact':sum(r['gold_score']==s['raw_score'] for r,s in items)/len(items) if items else None}
    summary['exact']=sum(r['gold_score']==s['raw_score'] for r,s in pairs)/len(pairs) if pairs else None
    threshold=rubric['features'][name]['success_threshold']
    summary['presence_threshold']=threshold
    summary['presence_agreement']=sum((r['gold_score']>=threshold)==s['success'] for r,s in pairs)/len(pairs) if pairs else None
    summary['errors']=[{'prompt_id':r['prompt_id'],'reference':r['gold_score'],'judge':s['raw_score'],'text':r['text']} for r,s in pairs if r['gold_score']!=s['raw_score']]
    if pairs:
        N=len(pairs);gold=Counter(r['gold_score'] for r,s in pairs);pred=Counter(s['raw_score'] for r,s in pairs); maximum=pairs[0][1]['scale_max']
        observed=sum(((r['gold_score']-s['raw_score'])/maximum)**2 for r,s in pairs)/N
        expected=sum(ng*np*((g-p)/maximum)**2 for g,ng in gold.items() for p,np in pred.items())/N**2
        summary['quadratic_kappa']=1-observed/expected if expected else None
        summary['complete_distributions']=sum(s['score_distribution']['complete'] for r,s in pairs)
        summary['distributions_with_observed_labels']=sum(bool(s['score_distribution']['observed_label_probabilities']) for r,s in pairs)
        summary['pilot_gate']=len(pairs)==len(ref) and summary['obvious']['exact']>=.90 and summary['hard']['exact']>=.75 and summary['quadratic_kappa']>=.60
        rep_path=ROOT/'runs/calibration'/f'{name}-repeat'/'scores.jsonl'
        reps=[json.loads(l) for l in rep_path.read_text(encoding='utf-8').splitlines()] if rep_path.exists() else []
        summary['repeats']={'n':len(reps),'matched':sum(scores[r['answer_id'].removesuffix('-repeat')]['raw_score']==r['raw_score'] for r in reps)}
    out['features'][name]=summary
raws=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'runs').rglob('raw/*.json')]
out['api']={'raw_requests':len(raws),'cost':sum((r.get('usage') or {}).get('cost',0) for r in raws),'reasoning_tokens':sum(((r.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens',0) for r in raws),'models':sorted({r.get('model','') for r in raws}),'providers':sorted({str(r.get('provider')) for r in raws})}
(ROOT/'calibration/summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
