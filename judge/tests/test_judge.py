import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ready_judge import core


def choice(digit='2', maximum=4):
    weights = [0.1] * (maximum + 1)
    weights[int(digit)] = 0.5
    return {'finish_reason': 'stop', 'message': {'content': digit},
            'logprobs': {'content': [{'token': digit, 'top_logprobs': [
                {'token': str(i), 'logprob': math.log(v)} for i, v in enumerate(weights)]}]}}


class JudgeTests(unittest.TestCase):
    def test_scales_and_probabilities(self):
        for maximum in (2, 3, 4):
            for score in range(maximum + 1):
                self.assertEqual(core.normalized(score, maximum), 100 * score / maximum)
            result = core.parse_choice(choice('2', maximum), maximum)
            self.assertTrue(result['score_distribution']['complete'])
            self.assertEqual(len(result['score_distribution']['probabilities']), maximum + 1)
            self.assertAlmostEqual(sum(result['score_distribution']['probabilities'].values()), 1)
        self.assertEqual(core.normalized(1, 2), core.normalized(2, 4))

    def test_missing_logprobs_not_invented(self):
        c = choice('2', 3)
        c['logprobs']['content'][0]['top_logprobs'].pop()
        r = core.parse_choice(c, 3)
        self.assertEqual(r['raw_score'], 2)
        self.assertIsNone(r['score_distribution']['probabilities'])
        self.assertIsNone(r['score_distribution']['expected_normalized_score_pct'])
        self.assertEqual(r['score_distribution']['missing_labels'], ['3'])
        self.assertIsNotNone(r['score_distribution']['available_expected_normalized_score_pct'])
        self.assertIn('2', r['score_distribution']['observed_label_probabilities'])

    def test_bad_output_and_invalid_scores(self):
        for text, finish in [('4', 'stop'), (' 2', 'stop'), ('2.', 'stop'), ('2', 'length')]:
            with self.assertRaises(ValueError):
                core.parse_choice({'finish_reason': finish, 'message': {'content': text}}, 3)
        with self.assertRaises(ValueError):
            core.normalized(True, 4)

    def test_logprob_conflict(self):
        c = choice('2', 4)
        c['logprobs']['content'][0]['top_logprobs'][0]['logprob'] = math.log(0.6)
        with self.assertRaises(ValueError):
            core.parse_choice(c, 4)

    def test_blinding_and_model_settings(self):
        cfg, _, prompts = core.configuration()
        p = core.payload({'scenario': 'Explain.', 'text': 'Hello.', 'method': 'GDN-SECRET'}, cfg, prompts['french'], 3)
        self.assertNotIn('GDN-SECRET', json.dumps(p))
        self.assertTrue(p['logprobs'])
        self.assertEqual(p['reasoning'], {'enabled': False})
        self.assertFalse(p['provider']['allow_fallbacks'])

    def test_resume_and_lock_without_api(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'input.jsonl'
            source.write_text(json.dumps({'prompt_id': 'p', 'answer_id': 'a', 'scenario': 'S', 'text': 'T'}) + '\n', encoding='utf-8')
            out = root / 'result'
            with patch('ready_judge.core.request_score') as request:
                r = core.evaluate(source, out, ['french'], run=False)
                self.assertEqual(r['expected'], 1)
                request.assert_not_called()
                core.evaluate(source, out, ['french'], run=False)
                with self.assertRaises(ValueError):
                    core.evaluate(source, out, ['numbered'], run=False)
                self.assertFalse((out / 'evaluation.lock').exists())
            source.write_text(source.read_text(encoding='utf-8') * 2, encoding='utf-8')
            with self.assertRaises(ValueError):
                core.input_rows(source)

    def test_raw_artifacts_never_contain_headers(self):
        cfg, rubric, prompts = core.configuration()
        raw = {'id':'fake', 'model':cfg['model'], 'provider':'CoreWeave',
               'choices':[choice('2', 3)], 'usage':{'completion_tokens_details':{'reasoning_tokens':0}}}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self): return json.dumps(raw).encode()
        class Opener:
            def open(self, *args, **kwargs): return Response()
        with tempfile.TemporaryDirectory() as d, patch('ready_judge.core.build_opener', return_value=Opener()):
            result = core.request_score({'task_id':'t','prompt_id':'p','answer_id':'a','feature':'french','scenario':'S','text':'T'},cfg,prompts['french'],rubric['features']['french'],'fake-private-secret',Path(d))
            self.assertEqual(result['normalized_score_pct'], 200/3)
            self.assertTrue(result['success'])
            self.assertNotIn('fake-private-secret', next((Path(d)/'raw').glob('*')).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
