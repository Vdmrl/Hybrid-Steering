import unittest
from types import SimpleNamespace

import torch

from gdn_interp import GDNRunner, add_delta, difference_states, truncate_svd


class Tokenizer:
    def __init__(self) -> None:
        self.add_special_tokens: bool | None = None

    def __call__(self, texts: list[str], **kwargs: object) -> SimpleNamespace:
        self.add_special_tokens = kwargs["add_special_tokens"]
        return SimpleNamespace(
            to=lambda device: SimpleNamespace(
                input_ids=torch.tensor([[1, 2], [0, 3]]), attention_mask=torch.tensor([[1, 1], [0, 1]])
            )
        )


class SteeringTest(unittest.TestCase):
    def test_truncate_svd_preserves_requested_rank(self) -> None:
        matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0]))
        truncated = truncate_svd(matrix, rank=1)
        self.assertEqual(torch.linalg.matrix_rank(truncated).item(), 1)
        self.assertTrue(torch.allclose(truncated, torch.diag(torch.tensor([4.0, 0.0, 0.0]))))

    def test_difference_states_and_zero_rank(self) -> None:
        first = {2: torch.ones(1, 2, 2)}
        second = {2: torch.zeros(1, 2, 2)}
        self.assertTrue(torch.equal(difference_states(first, second, rank=0)[2], first[2]))
        self.assertTrue(torch.equal(difference_states(first, second, rank=None)[2], first[2]))

    def test_normalized_delta_matches_current_state_norm(self) -> None:
        state = torch.tensor([[[[3.0, 4.0], [0.0, 0.0]]]])
        correction = add_delta(state, torch.ones(1, 2, 2), 1.0)
        self.assertTrue(torch.allclose(torch.linalg.matrix_norm(correction), torch.tensor([[5.0]])))

    def test_normalized_delta_is_raw_at_zero_state(self) -> None:
        state = torch.zeros(1, 1, 2, 2)
        delta = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        correction = add_delta(state, delta, 0.5)
        torch.testing.assert_close(correction, delta[None] * 0.5)

    def test_prompt_inputs_do_not_duplicate_chat_template_special_tokens(self) -> None:
        tokenizer = Tokenizer()
        input_ids, attention_mask = GDNRunner._inputs(SimpleNamespace(tokenizer=tokenizer, device="cpu"), ["one", "two"])
        self.assertFalse(tokenizer.add_special_tokens)
        self.assertTrue(torch.equal(input_ids[1], torch.tensor([0, 3])))
        self.assertTrue(torch.equal(attention_mask[1], torch.tensor([0, 1])))


if __name__ == "__main__":
    unittest.main()
