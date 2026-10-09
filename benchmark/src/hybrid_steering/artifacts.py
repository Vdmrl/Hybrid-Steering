"""Concept/method/protocol result store. Strengths append; incompatible protocols separate."""
import fcntl
import hashlib
import json
import re
from pathlib import Path

from hybrid_steering.judge.core import fingerprint, read_jsonl, write


def slug(value: object) -> str:
    return re.sub(r'[^a-z0-9_.-]+','-',str(value).lower()).strip('-')


def profile(manifest: dict, row: dict) -> dict:
    identity=manifest['identity'];config=identity['config'];bench=identity.get('benchmark',{})
    concept=config['concept'];method=row['method']
    assert method=='baseline' or method=='residual' or re.fullmatch(r'gdn_(full|rank\d+|clamp_rank\d+)',method)
    direction=identity.get('direction_sha256',identity.get('source_direction_sha256'))
    assert direction and manifest.get('model_revision'), 'Missing direction/model provenance'
    dataset=identity.get('dataset_sha256')
    assert dataset, 'Missing dataset identity; keep legacy, do not guess'
    schedule=row.get('schedule')
    if not schedule:
        schedule='none' if method=='baseline' else 'every_token' if method=='residual' else (
            identity.get('clamp_schedule','clamp') if method.startswith('gdn_clamp') else 'once')
    normalization=identity.get('normalization_by_method',{}).get(method)
    if normalization is None:
        normalization=(identity.get('normalization','per-head Frobenius to full-rank')
                       if method.startswith('gdn_clamp') and (identity.get('normalization') or bench.get('normalize_rank1_to_full'))
                       else 'none')
    # Do not key by run date, strengths, batch size, host, or output directory.
    # Preserve core code revisions: different intervention implementations are
    # not silently treated as the same scientific protocol.
    return dict(concept=concept,method=method,model=config['model'],model_revision=manifest['model_revision'],
                tokenizer_revision=manifest.get('tokenizer_revision'),thinking=row['thinking'],
                max_new_tokens=bench.get('max_new_tokens',identity.get('max_new_tokens',config['max_new_tokens'])),
                system=bench.get('system',config.get('system')),schedule=schedule,
                layer=row.get('layer',-1),layers=identity.get('layers',config.get('layers','recorded in source manifest')),
                heads=identity.get('heads',config.get('head_percentages')),normalization=normalization,
                direction_sha256=direction if method!='baseline' else None,dataset_sha256=dataset,
                core_code={k:v for k,v in identity.get('code',{}).items() if k in ('src/hybrid_steering/runner.py','src/hybrid_steering/state.py','src/hybrid_steering/residual.py')},
                evaluator_sha256=manifest.get('evaluator_sha256'),
                generation_options={k:config[k] for k in ('do_sample','temperature','top_p','seed') if k in config})


def location(root: Path, identity: dict) -> Path:
    model_name=slug(identity['model'].split('/')[-1])
    model_folder='9b' if model_name.endswith('9b') else '4b' if model_name.endswith('4b') else model_name
    method=identity['method'].replace('gdn_full','gdn-add-full').replace('gdn_clamp_rank','gdn-clamp-rank').replace('gdn_rank','gdn-add-rank')
    name=f"{slug(identity['schedule'])}_layer{identity['layer']}_thinking-{'on' if identity['thinking'] else 'off'}_tokens{identity['max_new_tokens']}"
    name+=f"_norm-{'off' if identity['normalization']=='none' else 'on'}_protocol-{fingerprint(identity)[:10]}"
    return root/model_folder/slug(identity['concept'])/method/model_name/name


def merge(folder: Path, identity: dict, rows: list, source: dict) -> int:
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=folder/'manifest.json'
        state=json.loads(manifest.read_text()) if manifest.exists() else dict(identity=identity,sources=[])
        assert state['identity']==identity, 'Protocol mismatch'
        path=folder/'answers.jsonl'
        old=read_jsonl(path) if path.exists() else []
        def key(r):return fingerprint([r['method'],r.get('layer',-1),float(r['strength']),r.get('key',r.get('id'))])
        saved={key(r):r for r in old}
        assert len(saved)==len(old), 'Existing duplicate keys'
        count=0
        for row in rows:
            k=key(row)
            if k in saved:
                assert all(saved[k].get(f)==row.get(f) for f in ('prompt','response','model','thinking')), 'Conflicting answer; no overwrite'
                # Scored rows may add benchmark fields; contradictory scores fail.
                for f,v in row.items():
                    if f.startswith('ifeval_'):
                        assert f not in saved[k] or saved[k][f]==v, 'Conflicting benchmark score'
                        saved[k][f]=v
            else:
                saved[k]={**row,'concept':identity['concept']};count+=1
        state['sources']=sorted({json.dumps(s,sort_keys=True) for s in state['sources']}|{json.dumps(source,sort_keys=True)})
        state['sources']=[json.loads(s) for s in state['sources']]
        state['n_answers']=len(saved)
        state['strengths']=sorted({float(r['strength']) for r in saved.values()})
        write(path,list(saved.values()));write(manifest,state)
        return count
