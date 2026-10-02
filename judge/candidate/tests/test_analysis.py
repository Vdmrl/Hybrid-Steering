import unittest
from analysis import analyze


class AnalysisTests(unittest.TestCase):
    def test_paired_difference_and_missing_cells(self):
        scores=[]; mapping=[]
        for p in ('p1','p2','p3'):
            for method in ('caa','gdn'):
                aid=p+method
                mapping.append({'prompt_id':p,'answer_id':aid,'method':method,'condition':'NFT1'})
                scores.append({'prompt_id':p,'answer_id':aid,'feature':'numbered','normalized_score_pct':100 if method=='gdn' else 50,'success':method=='gdn'})
        result=analyze(scores,mapping,draws=100)
        d=next(x for x in result['paired_differences'] if x['endpoint']=='mean_normalized_score_pct')
        self.assertEqual(d['difference_percentage_points'],-50)
        self.assertEqual(d['paired_ci95'],[-50,-50])
        with self.assertRaises(ValueError):analyze(scores[:-1],mapping,draws=100)
        with self.assertRaises(ValueError):analyze([s for s in scores if s['answer_id'].endswith('gdn')],mapping,draws=100)
        with self.assertRaises(ValueError):analyze([s for s in scores if s['prompt_id']!='p3'],mapping,draws=100)


if __name__=='__main__':unittest.main()
