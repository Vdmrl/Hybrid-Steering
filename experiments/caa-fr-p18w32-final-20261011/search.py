"""Finite follow-up tasks: frozen candidates, controls, no Judge/automatic success claim."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import os
import uuid

from layerwise_composition import rss_coefficients_by_layer


def execute(model, tokenizer, device, original, args, generate):
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from temporal_hook import hook_for
    task=json.loads(args.config.read_text())
    assert hashlib.sha256(Path(__file__).with_name('frozen-config.json').read_bytes()).hexdigest()==task['frozen_config_sha256']
    if task['method']=='extract':
        from extract_feature import extract
        root=args.output.parent.parent
        extract(model,tokenizer,device,original,root)
        if dist.get_rank()==0:
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/'manifest.json').write_text(json.dumps({'status':'COMPLETE_PENDING_MANUAL_REVIEW','task':task,'direction_manifest_sha256':hashlib.sha256((root/'local/directions/manifest.json').read_bytes()).hexdigest()},indent=2))
        return
    for feature,source in task['sources'].items():
        if 'direction_file' in source:
            root=args.output.parent.parent
            manifest=json.loads((root/'local/directions/manifest.json').read_text())
            assert manifest['provenance'][feature]['sha256']==source['expected_pairs_sha256']
            file=source['direction_file']
            task['sources'][feature]={'path':str(root/'local/directions'/file),'sha256':manifest['directions_sha256'][file]}
    method=task['method']; bank=task['prompts']; jobs=task['jobs']
    assert method == 'caa' and bank and jobs
    assert len({p['id'] for p in bank})==len(bank)
    assert len({j['id'] for j in jobs})==len(jobs)
    tensors={}
    for feature,source in task['sources'].items():
        p=Path(source['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=source['sha256']: raise ValueError('changed direction '+feature)
        values={int(k.removeprefix('layer_')):v.float() for k,v in load_file(str(p)).items()}
        assert all(torch.isfinite(v).all() for v in values.values())
        tensors[feature]=values
    rank=dist.get_rank() if dist.is_initialized() else 0;stream=None
    if dist.is_initialized(): dist.barrier()
    mode=args.runtime_mode
    session=os.environ['STEERING_LOAD_SESSION']
    previous=[]
    raw_path=args.output/'generations.jsonl'
    if raw_path.exists(): previous=[json.loads(line) for line in raw_path.read_text().splitlines()]
    completed={r['answer_id'] for r in previous if r['phase']=='screen'}
    config_hash=hashlib.sha256(args.config.read_bytes()).hexdigest()
    if (args.output/'manifest.json').exists():
        saved=json.loads((args.output/'manifest.json').read_text())
        if saved['config_sha256'] != config_hash: raise ValueError('resume config changed')
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=True)
        stream=(args.output/'generations.jsonl').open('a',encoding='utf-8')
        manifest={'status':'RUNNING','task':task,'config_sha256':config_hash,
                  'dtype':'bfloat16','tp_size':2 if mode=='native-tp2' else 1,'offload':mode=='single-offload','runtime_mode':mode,'load_session':session,'previous_screen_answers':len(completed),'device_map':getattr(model,'hf_device_map',None),'runtime_segments':sorted({r.get('runtime_mode','unknown') for r in previous}|{mode}),'seed':20260911,'judge_api_used':False,
                  'expected_screen_answers':len(bank)*len(jobs),'frozen_config_sha256':task['frozen_config_sha256'],'temporal_hook_sha256':hashlib.sha256(Path(__file__).with_name('temporal_hook.py').read_bytes()).hexdigest(), 'hook_sha256':hashlib.sha256(Path('caa_hook.py').read_bytes()).hexdigest() if Path('caa_hook.py').exists() else hashlib.sha256(Path(__file__).with_name('caa_hook.py').read_bytes()).hexdigest()}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def answer(prompt,job=None,zero=False,phase='screen'):
        if mode=='single-offload' and (args.output/'upgrade-request.json').exists():
            raise SystemExit(75)
        audit=[];handles=[]
        if job:
            components={}; coordinate={}
            for f,v in job['features'].items():
                for i in v['layers']:
                    raw=tensors[f][i]; assert raw.ndim==1 and raw.norm()>0
                    unit=raw/raw.norm(); gain=v['gain']/(len(v['layers'])**.5)
                    delta=unit*(0 if zero else gain)
                    components.setdefault(i,[]).append((delta,v.get("max_calls"),v.get("min_calls",1)))
                    coordinate.setdefault(i,[]).append((unit,gain,v.get('beta',.25)))
            if job.get('intervention','additive')=='coordinate_clamp':
                from caa_clamp import hook_for as clamp_for
                handles=[model.model.language_model.layers[i].register_forward_hook(
                    clamp_for([x[0] for x in rows],[x[1] for x in rows],[x[2] for x in rows],audit,disabled=zero)) for i,rows in sorted(coordinate.items())]
            else:
                handles=[model.model.language_model.layers[i].register_forward_hook(hook_for(v,audit)) for i,v in sorted(components.items())]
        started=time.monotonic()
        try:tokens,text=generate(model,tokenizer,prompt['text'],task['max_new_tokens'],device,original)
        finally:
            for h in handles:h.remove()
        if dist.is_initialized():
            gathered=[None,None];dist.all_gather_object(gathered,{'tokens':tokens,'calls':len(audit)})
            if gathered[0]!=gathered[1]:raise RuntimeError('TP ranks disagree')
        if job and not audit:raise RuntimeError('intervention never fired')
        r={'answer_id':task['id']+':'+prompt['id']+':'+(job['id'] if job else phase)+(':zero' if zero else ''),
           'prompt_id':prompt['id'],'prompt':prompt['text'],'phase':phase,'job':job,'zero':zero,
           'response':text,'token_ids':tokens,'finish_reason':'length' if len(tokens)==task['max_new_tokens'] else 'eos',
           'seconds':time.monotonic()-started,'audit_first':audit[:2],
           'runtime_mode':mode,'load_session':session,'gpu_indices':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_uuids':os.environ.get('STEERING_GPU_UUIDS'),'offload':mode=='single-offload','tp_size':2 if mode=='native-tp2' else 1}
        if phase!='screen': r['answer_id']+=':'+session+':'+str(len(previous))
        previous.append(r)
        if stream:stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno());print('GENERATED',phase,prompt['id'],job['id'] if job else 'baseline',len(tokens),flush=True)
        return tokens
    try:
        baselines={p['id']:answer(p,phase='baseline') for p in bank}
        if rank==0:
            manifest['cross_runtime_baseline_comparison']=[
                {'prompt_id':r['prompt_id'],'previous_session':r['load_session'],
                 'previous_runtime':r['runtime_mode'],'token_equal':r['token_ids']==baselines[r['prompt_id']]}
                for r in previous if r['phase']=='baseline' and r['load_session']!=session]
            manifest['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        controls=[]
        for feature in task['sources']:
            controls.append(next(j for j in jobs if feature in j['features']))
        for control in controls:
            for p in bank[:2]:
                if answer(p,control,zero=True,phase='zero')!=baselines[p['id']]:raise RuntimeError('zero parity failure')
        if rank==0:print('ZERO_HOOK_PARITY_PASS',flush=True)
        for job in jobs:
            for p in bank:
                answer_id=task['id']+':'+p['id']+':'+job['id']
                if answer_id not in completed: answer(p,job)
            for p in bank[:1]:
                if answer(p,phase='replay')!=baselines[p['id']]:raise RuntimeError('baseline replay failure')
            if rank==0:print('JOB_DONE',job['id'],flush=True)
        if rank==0:
            assert len({r['answer_id'] for r in previous if r['phase']=='screen'})==len(bank)*len(jobs)
            manifest['status']='COMPLETE_PENDING_MANUAL_REVIEW'
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print('COMPLETE_PENDING_MANUAL_REVIEW',flush=True)
    finally:
        if stream:stream.close()
