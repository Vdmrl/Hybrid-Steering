"""Explicit pilot calibration command; no paid requests without --run."""
import argparse
import getpass
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ready_judge.core import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    key = getpass.getpass('OpenRouter key (not saved): ') if args.run else None
    for name in ('numbered','french','complexity','fairy_tale'):
        path = ROOT/'calibration'/f'{name}-blind.jsonl'
        result = evaluate(path,ROOT/'runs/calibration'/name,[name],args.run,4,key)
        print(name,json.dumps(result),flush=True)
        if args.run and not result['complete']:
            raise SystemExit('Calibration stopped on missing scores; inspect raw/errors')
        records = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()][::4]
        for row in records:
            row['answer_id'] += '-repeat'
        repeat = ROOT/'calibration'/f'{name}-repeat.jsonl'
        repeat.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        result=evaluate(repeat,ROOT/'runs/calibration'/f'{name}-repeat',[name],args.run,4,key)
        print(name+'-repeat',json.dumps(result),flush=True)
    key=None


if __name__=='__main__':main()
