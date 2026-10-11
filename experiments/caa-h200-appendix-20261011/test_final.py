import json
from pathlib import Path
import unittest


class FinalContracts(unittest.TestCase):
    def test_windows_preserve_continuous_features(self):
        import torch
        from temporal_hook import hook_for
        x=torch.zeros(1,1,4,dtype=torch.bfloat16)
        a=torch.ones(4); audit=[]
        hook=hook_for([(a,2),(a*2,3),(a*4,None)],audit)
        bulk=torch.zeros(1,8,4)
        self.assertIs(hook(None,None,bulk),bulk)
        for expected in [7,7,6,4,4]:
            self.assertTrue(torch.equal(hook(None,None,x),x+expected))
        self.assertEqual([v['active_components'] for v in audit],[3,3,2,1,1])
        self.assertTrue(torch.equal(hook_for([(a*0,2),(a*0,None)],[])(None,None,x),x))
        short=hook_for([(a,1)],[])
        short(None,None,x)
        self.assertIs(short(None,None,x),x)

    def test_matched_disjoint_design(self):
        base=Path(__file__).parent
        tasks=[json.loads(p.read_text()) for p in sorted((base/'tasks').glob('*.json'))]
        ids=[p['id'] for t in tasks for p in t['prompts']]
        self.assertEqual(len(ids),64);self.assertEqual(len(set(ids)),32)
        self.assertEqual(sum(len(t['prompts'])*len(t['jobs']) for t in tasks)+32,2560)
        self.assertEqual(len({j['id'] for t in tasks for j in t['jobs']}),79)
        for t in tasks:
            self.assertEqual(t['resource_class'],'final')
            for j in t['jobs']:
                for name,v in j['features'].items():
                    self.assertEqual(v.get('max_calls'),28 if name=='numbered' else 24 if name=='probabilistic_framing' else None)
                    self.assertNotIn('beta',v)
                    frozen=json.loads((base/'frozen-config.json').read_text())['features'][name]
                    self.assertEqual(v['layers'],frozen['layers'])
                    self.assertEqual(v['gain'],frozen['gain']*j['lambda'])


if __name__=='__main__':unittest.main()
