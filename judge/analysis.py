"""Offline balanced-cohort percentages and prompt-paired bootstrap intervals."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    values=sorted(values); position=(len(values)-1)*fraction
    lo=int(position); hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def analyze(scores, mapping, draws=2000, seed=20261002):
    if draws < 100:
        raise ValueError('At least100 bootstrap draws required')
    bindings={}
    for r in mapping:
        key=r['prompt_id'],r['answer_id']
        if key in bindings:
            raise ValueError('Duplicate private binding')
        bindings[key]=r
    cells=defaultdict(dict)
    for s in scores:
        b=bindings.get((s['prompt_id'],s['answer_id']))
        if b is None:
            raise ValueError('Unmapped answer')
        key=s['feature'], b['method'], b['condition']
        if s['prompt_id'] in cells[key]:
            raise ValueError('One answer per prompt/method/condition/feature is required')
        cells[key][s['prompt_id']]=(s['normalized_score_pct'],100.0*bool(s['success']))
    if not cells:
        raise ValueError('No scores')
    prompts=sorted(next(iter(cells.values())))
    if len(prompts)<2 or any(set(v)!=set(prompts) for v in cells.values()):
        raise ValueError('Require at least two identical prompt sets in all compared cells; do not silently drop missing scores')
    # Mapping defines the expected groups: an entirely missing cell must also be rejected.
    expected_groups={(r['method'],r['condition']) for r in mapping}
    for (feature, method, condition), cell in cells.items():
        expected_prompts={r['prompt_id'] for r in mapping if r['method']==method and r['condition']==condition}
        if set(cell)!=expected_prompts:
            raise ValueError('Expected prompt missing; mapping and score coverage differ')
    for feature in {k[0] for k in cells}:
        if {(k[1],k[2]) for k in cells if k[0]==feature}!=expected_groups:
            raise ValueError('Entire method/condition cell missing')
    rng=random.Random(seed)
    resamples=[[rng.randrange(len(prompts)) for _ in prompts] for _ in range(draws)]
    values={k:[v[p] for p in prompts] for k,v in cells.items()}
    summaries=[]; means={}
    for key, v in values.items():
        for endpoint,column in (('mean_normalized_score_pct',0),('success_rate_pct',1)):
            observed=sum(x[column] for x in v)/len(v)
            boots=[sum(v[i][column] for i in ids)/len(v) for ids in resamples]
            means[(key,endpoint)]=(observed,boots)
            summaries.append({'feature':key[0],'method':key[1],'condition':key[2],'endpoint':endpoint,'n_prompts':len(v),'estimate':observed,'ci95':[percentile(boots,.025),percentile(boots,.975)]})
    differences=[]
    for feature, condition in sorted({(k[0],k[2]) for k in values}):
        methods=sorted(k[1] for k in values if k[0]==feature and k[2]==condition)
        for j,left in enumerate(methods):
            for right in methods[j+1:]:
                for endpoint in ('mean_normalized_score_pct','success_rate_pct'):
                    a, ab=means[((feature,left,condition),endpoint)]
                    b, bb=means[((feature,right,condition),endpoint)]
                    bootstrap=[x-y for x,y in zip(ab,bb)]
                    differences.append({'feature':feature,'condition':condition,'left':left,'right':right,'endpoint':endpoint,'difference_percentage_points':a-b,'paired_ci95':[percentile(bootstrap,.025),percentile(bootstrap,.975)]})
    return {'status':'exploratory intervals; not multiple-comparison corrected','bootstrap_draws':draws,'seed':seed,'sampling_unit':'whole prompt_id; common resamples across all cells','n_prompts':len(prompts),'cells':summaries,'paired_differences':differences}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scores',required=True)
    parser.add_argument('--mapping',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--draws',type=int,default=2000)
    args=parser.parse_args()
    read=lambda p:[json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
    result=analyze(read(args.scores),read(args.mapping),args.draws)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'n_prompts':result['n_prompts'],'cells':len(result['cells'])}))


if __name__=='__main__':main()
