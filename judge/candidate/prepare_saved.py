"""Convert saved9B/4B generations to blind input with separate private metadata."""
import argparse
import json
from pathlib import Path
from ready_judge.core import fingerprint


def convert(rows, method_label=None):
    blind=[]; private=[]; seen=set(); scenarios={}
    for r in rows:
        pid=r.get('prompt_id')
        scenario=r.get('scenario',r.get('prompt'))
        text=r.get('text',r.get('response'))
        if not isinstance(pid,str) or not pid or not isinstance(scenario,str) or not isinstance(text,str):
            raise ValueError('Saved row needs prompt_id, prompt/scenario, response/text')
        method=r.get('method',method_label)
        if not isinstance(method,str) or not method:
            raise ValueError('Supply --method-label if saved data has no method')
        condition=r.get('condition')
        if not condition:
            condition='+'.join(sorted(r.get('active_features',[]))) or 'baseline'
            condition+=f'@lambda={r.get("lambda",1)}'
        setting=r.get('setting',{})
        if setting:
            condition+='@'+json.dumps(setting,sort_keys=True,separators=(',',':'))
        aid=r.get('answer_id') or fingerprint({'prompt_id':pid,'method':method,'condition':condition,'seed':r.get('seed')})[:24]
        if (pid,aid) in seen:
            raise ValueError('Duplicate answer identity; source must distinguish seeds/configurations')
        if pid in scenarios and scenarios[pid]!=scenario:
            raise ValueError('Same prompt_id has different scenarios')
        scenarios[pid]=scenario;seen.add((pid,aid))
        blind.append({'prompt_id':pid,'answer_id':aid,'scenario':scenario,'text':text})
        private.append({'prompt_id':pid,'answer_id':aid,'method':method,'condition':condition,'finish_reason':r.get('finish_reason'),'lambda':r.get('lambda'),'setting':setting})
    return blind,private


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,nargs='+')
    parser.add_argument('--blind',required=True)
    parser.add_argument('--mapping',required=True)
    parser.add_argument('--method-label')
    args=parser.parse_args()
    if Path(args.blind).resolve()==Path(args.mapping).resolve():
        parser.error('Blind and private mapping paths must differ')
    if any(Path(p).exists() for p in (args.blind,args.mapping)):
        parser.error('Refusing to overwrite existing export')
    rows=[json.loads(l) for p in args.input for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
    blind,mapping=convert(rows,args.method_label)
    for path,items in ((args.blind,blind),(args.mapping,mapping)):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():raise ValueError('Refusing to overwrite existing export')
        p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
    print(json.dumps({'answers':len(blind),'prompt_ids':len({r['prompt_id'] for r in blind})}))


if __name__=='__main__':main()
