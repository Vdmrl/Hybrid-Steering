import unittest

import torch

from circuit_lens import comp_score
from gdn import GatedDeltaNet
from induction_lens import head_scores, repeated_token_batch
from model import ModelSpec, build_model, sinusoidal_positions
from skip_trigram_lens import find_skip_trigrams
from train import cosine_schedule, log_spaced


class ToyTokenizer:
    all_special_ids = []
    eos_token_id = 0

    @staticmethod
    def decode(token_ids):
        return f"t{token_ids[0]}"


class ToyModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ModelSpec(d_model=16, n_heads=2, d_head=8, n_ctx=8, seed=1)
        cls.model = build_model(cls.spec, 32)

    def test_model_is_attention_only_and_runnable(self):
        tokens = torch.randint(0, 32, (2, self.spec.n_ctx))
        logits = self.model(tokens)
        self.assertEqual(tuple(logits.shape), (2, self.spec.n_ctx, 32))
        self.assertFalse(self.model.pos_embed.W_pos.requires_grad)
        self.assertEqual(self.model.cfg.n_layers, 1)
        self.assertTrue(self.model.cfg.attn_only)

    def test_builds_two_layer_model(self):
        spec = ModelSpec(d_model=16, n_heads=2, d_head=8, n_ctx=8, seed=1, layer_types=("attn", "attn"))
        model = build_model(spec, 32)
        tokens = torch.randint(0, 32, (2, spec.n_ctx))
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, spec.n_ctx, 32))
        self.assertEqual(model.cfg.n_layers, 2)
        self.assertTrue(model.cfg.attn_only)

    def test_gdn_and_hybrid_layer_orders_are_hookable(self):
        tokens = torch.randint(0, 32, (2, self.spec.n_ctx))
        for layers in (("gdn",), ("gdn", "attn"), ("attn", "gdn"), ("gdn", "gdn")):
            model = build_model(
                ModelSpec(d_model=16, n_heads=2, d_head=8, n_ctx=8, seed=1, layer_types=layers),
                32,
            )
            logits, cache = model.run_with_cache(tokens)
            self.assertEqual(tuple(logits.shape), (2, 8, 32))
            logits.mean().backward()
            for index, layer in enumerate(layers):
                if layer == "gdn":
                    self.assertIn(f"blocks.{index}.gdn.hook_state", cache)

    def test_forced_fla_backend_rejects_cpu(self):
        gdn = GatedDeltaNet(16, 2, 8, backend="fla")
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            gdn(torch.randn(1, 4, 16))

    def test_fixed_sinusoidal_positions(self):
        positions = sinusoidal_positions(3, 4)
        self.assertTrue(torch.equal(positions[0, 0::2], torch.zeros(2)))
        self.assertTrue(torch.equal(positions[0, 1::2], torch.ones(2)))

    def test_skip_trigram_lens_returns_each_head(self):
        sequences = [torch.arange(8), torch.arange(7, -1, -1)]
        rows, candidates = find_skip_trigrams(
            self.model,
            ToyTokenizer(),
            sequences,
            candidate_vocab=8,
            pairs_per_head=2,
            outputs_per_pair=2,
            skip_window=4,
        )
        self.assertEqual(len(candidates), 8)
        self.assertEqual(len(rows), 2 * 2 * 2 * 2)
        self.assertEqual({row["head"] for row in rows}, {0, 1})
        self.assertTrue(all("key_score" in row for row in rows))

    def test_schedule_warms_up_and_decays(self):
        schedule = cosine_schedule(100, 10, 0.1)
        self.assertLess(schedule(0), schedule(9))
        self.assertAlmostEqual(schedule(100), 0.1)

    def test_log_spaced_steps(self):
        steps = log_spaced(1000, 8)
        self.assertEqual(steps, sorted(set(steps)))
        self.assertEqual(steps[0], 1)
        self.assertEqual(steps[-1], 1000)

    def test_probe_seed_is_deterministic(self):
        sequences = [torch.randint(1, 32, (16,)) for _ in range(4)]
        first = repeated_token_batch(ToyTokenizer(), sequences, seq_len=5, batch=3, seed=7)
        second = repeated_token_batch(ToyTokenizer(), sequences, seq_len=5, batch=3, seed=7)
        self.assertTrue(torch.equal(first, second))

    def test_induction_score_index_arithmetic(self):
        n = 8
        induction_only = torch.zeros(1, 2 * n + 1, 2 * n + 1)
        for q in range(n + 1, 2 * n + 1):
            induction_only[0, q, q - (n - 1)] = 1.0
        scores = head_scores(induction_only, n)
        self.assertAlmostEqual(scores["induction"], 1.0)
        self.assertAlmostEqual(scores["duplicate"], 0.0)

        duplicate_only = torch.zeros(1, 2 * n + 1, 2 * n + 1)
        for q in range(n + 1, 2 * n + 1):
            duplicate_only[0, q, q - n] = 1.0
        scores = head_scores(duplicate_only, n)
        self.assertAlmostEqual(scores["duplicate"], 1.0)
        self.assertAlmostEqual(scores["induction"], 0.0)

    def test_comp_score_detects_composition(self):
        generator = torch.Generator().manual_seed(0)
        a = torch.randn(16, 4, generator=generator)
        b = torch.randn(16, 4, generator=generator)
        composed = comp_score(a @ b.T, b @ a.T)
        random_pair = comp_score(a @ b.T, torch.randn(16, 16, generator=generator))
        self.assertGreater(composed, random_pair)


if __name__ == "__main__":
    unittest.main()
