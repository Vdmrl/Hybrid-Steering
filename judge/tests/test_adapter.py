import unittest
from prepare_saved import convert


class AdapterTests(unittest.TestCase):
    def test_preserves_ids_and_no_method_in_blind(self):
        rows=[{'prompt_id':'p','answer_id':'a1','prompt':'P','response':'same text','method':'caa','active_features':[],'lambda':0}, {'prompt_id':'p','answer_id':'a2','prompt':'P','response':'same text','method':'gdn','active_features':[],'lambda':0}]
        blind,mapping=convert(rows)
        self.assertEqual(len(blind),2)
        self.assertEqual([r['answer_id'] for r in blind],['a1','a2'])
        self.assertNotIn('method',blind[0])
        self.assertEqual(mapping[1]['method'],'gdn')
        self.assertEqual(convert(rows[::-1])[0][0]['answer_id'],'a2')

    def test_synthesized_ids_independent_of_order(self):
        rows=[{'prompt_id':'p','prompt':'P','response':'R','condition':'french','setting':{'gain':g,'layer':8}} for g in (2,4)]
        a,_=convert(rows,'caa');b,_=convert(rows[::-1],'caa')
        self.assertEqual(a[0]['answer_id'],b[1]['answer_id'])
        self.assertNotEqual(a[0]['answer_id'],a[1]['answer_id'])
        with self.assertRaises(ValueError):convert(rows)


if __name__=='__main__':unittest.main()
