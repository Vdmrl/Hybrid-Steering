import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from gdn_interp import GDNRunner, GDNTrace, add_delta


def batch(prompts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=0, padding_side="left")
    return input_ids, input_ids.ne(0)


class GenerationTest(unittest.TestCase):
    def test_split_prefill_preserves_prompt_and_decode_logits(self) -> None:
        torch.manual_seed(42)
        torch.set_num_threads(2)
        config = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3]},
        )
        with patch.object(qwen, "FusedRMSNormGated", None):
            model = Qwen3_5ForCausalLM(config).eval()
        module = model.model.layers[0].linear_attn
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = qwen.torch_causal_conv1d_update
        module.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
        module.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
        runner = GDNRunner(model, SimpleNamespace(pad_token_id=0, eos_token_id=63), [0], {0: torch.ones(2, 8, 8)})
        prompts = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([8, 9, 10, 11, 12, 13, 14])]
        input_ids, attention_mask = batch(prompts)

        with torch.inference_mode():
            for position in (None, -1, 0, 1, 3):
                cache, logits, mask = runner.prefill(input_ids, attention_mask, position, 0.0)
                for index, prompt in enumerate(prompts):
                    reference = model(input_ids=prompt[None], use_cache=True)
                    torch.testing.assert_close(logits[index], reference.logits[0, -1], atol=2e-5, rtol=2e-5)

                output = model(input_ids=torch.tensor([[15], [15]]), attention_mask=torch.cat((mask, torch.ones_like(mask[:, :1])), 1), position_ids=mask.long().sum(-1)[:, None], past_key_values=cache)
                for index, prompt in enumerate(prompts):
                    reference = model(input_ids=torch.cat((prompt, torch.tensor([15])))[None])
                    torch.testing.assert_close(output.logits[index, -1], reference.logits[0, -1], atol=2e-5, rtol=2e-5)

            with patch.object(runner, "_inputs", return_value=(input_ids, attention_mask)), patch("gdn_interp.generation.add_delta", wraps=add_delta) as inject:
                baseline = runner.generate(["first", "second"], prompt_steer_position=None, max_new_tokens=8)
                zero = runner.generate(["first", "second"], scale=0.0, max_new_tokens=8)
                inject.assert_called_once()
                self.assertTrue(inject.call_args.args[2].eq(0).all())
                torch.testing.assert_close(zero, baseline, atol=0, rtol=0)

            _, zero_logits, _ = runner.prefill(input_ids, attention_mask, -1, 0.0)
            _, steered_logits, _ = runner.prefill(input_ids, attention_mask, -1, 0.5)
            self.assertTrue(torch.isfinite(steered_logits).all())
            self.assertGreater((steered_logits - zero_logits).abs().max().item(), 1e-5)

            for position in (-1, 0, 1, 3):
                runner.normalize = False
                _, batched, _ = runner.prefill(input_ids, attention_mask, position, torch.tensor([0.0, 0.1]))
                for index, prompt in enumerate(prompts):
                    _, single, _ = runner.prefill(prompt[None], torch.ones_like(prompt[None]), position, 0.1 * index)
                    torch.testing.assert_close(batched[index], single[0], atol=2e-5, rtol=2e-5)

            for mode in ("exact", "chunk"):
                trace = GDNTrace()
                _, batched, _ = runner.prefill(input_ids, attention_mask, -1, 0.1, mode, trace)
                for index, prompt in enumerate(prompts):
                    position = -1 if mode == "exact" else 0
                    cache, single, _ = runner.prefill(prompt[None], torch.ones_like(prompt[None]), position, 0.1)
                    torch.testing.assert_close(batched[index], single[0], atol=2e-5, rtol=2e-5)
                    indices = trace.indices[len(prompt)]
                    self.assertIn(index, indices.tolist())
                    torch.testing.assert_close(trace.states[len(prompt)][0][indices.eq(index)], cache.layers[0].recurrent_states[0], atol=2e-5, rtol=2e-5)

            with patch.object(model, "forward", wraps=model.forward) as forward:
                runner.prefill(input_ids, attention_mask, torch.tensor([1, 3]), 0.1)
                forward.assert_called_once()

            long_prompts = [torch.arange(length) % 62 + 1 for length in (69, 77)]
            long_input_ids, long_attention_mask = batch(long_prompts)
            trace = GDNTrace()
            _, batched, _ = runner.prefill(long_input_ids, long_attention_mask, -1, 0.1, "chunk", trace)
            self.assertEqual(trace.prompt_positions.tolist(), [56, 64])
            for index, prompt in enumerate(long_prompts):
                _, single, _ = runner.prefill(
                    prompt[None], torch.ones_like(prompt[None]), int(trace.prompt_positions[index]), 0.1
                )
                torch.testing.assert_close(batched[index], single[0], atol=2e-5, rtol=2e-5)

            runner.normalize = True
            with patch.object(runner, "_inputs", return_value=(input_ids, attention_mask)):
                plain = runner.generate(["first", "second"], scale=torch.tensor([0.0, 0.1]), max_new_tokens=4)
                trace = GDNTrace()
                traced = runner.generate(["first", "second"], scale=torch.tensor([0.0, 0.1]), max_new_tokens=4, trace=trace)
                torch.testing.assert_close(plain, traced, atol=0, rtol=0)
                self.assertTrue(trace.outputs and trace.states)
                self.assertTrue(trace.prefill_outputs and trace.prefill_indices)

                for period, steps in ((2, None), (None, {1})):
                    trace = GDNTrace()
                    plain = runner.generate(["first", "second"], scale=0.1, generation_steer_period=period, generation_steer_steps=steps, max_new_tokens=4)
                    traced = runner.generate(["first", "second"], scale=0.1, generation_steer_period=period, generation_steer_steps=steps, max_new_tokens=4, trace=trace)
                    torch.testing.assert_close(plain, traced, atol=0, rtol=0)

                model.generation_config.eos_token_id = [int(plain[0, 0]), 63]
                stopped = runner.generate(["first", "second"], scale=0.1, max_new_tokens=4)
                self.assertTrue(stopped[0, 1:].eq(runner.tokenizer.pad_token_id).all())


if __name__ == "__main__":
    unittest.main()
