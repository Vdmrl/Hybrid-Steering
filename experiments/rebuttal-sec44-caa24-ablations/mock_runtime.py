"""Small CPU model for checking runner behavior without checkpoint weights."""

from types import SimpleNamespace

import torch
from build_job_configs import PACKAGE, condition, read_rows
from interventions import FEATURES, LAYERS
from torch import nn


class MockDecoder(nn.Module):
    def forward(self, x):
        return x + 0.001


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(MockDecoder() for _ in range(32))
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=self.blocks))
        self.generation_config = SimpleNamespace(eos_token_id=0)
        self.forward_records = []

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        use_cache,
        logits_to_keep,
        past_key_values=None,
    ):
        length = input_ids.shape[1]
        if past_key_values is None:
            assert length > 1
            cache = SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        recurrent_states=[torch.zeros(1, 2, 2, 2)],
                        conv_states=torch.zeros(1, 8),
                    )
                    if i in LAYERS
                    else SimpleNamespace(keys=torch.zeros(1, 2), values=torch.ones(1, 2))
                    for i in range(32)
                ],
                count=0,
            )
        else:
            assert length == 1
            cache = past_key_values
        assert attention_mask.shape[1] == cache.count + length
        assert position_ids[0].tolist() == list(range(cache.count, cache.count + length))
        cache.count += length
        assert use_cache and logits_to_keep == 1
        x = torch.nn.functional.one_hot(input_ids, num_classes=8).float()
        incoming = cache.layers[0].recurrent_states[0].clone()
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i in LAYERS:
                state = cache.layers[i].recurrent_states[0]
                state.mul_(0.90).add_(x[:, -1, :].reshape_as(state) * 0.01)
        self.forward_records.append({"length": length, "count": cache.count, "incoming": incoming})
        logits = x[:, -1:, :].clone()
        logits[:, :, 0] = -1000
        logits[:, :, 1] += 10
        return SimpleNamespace(past_key_values=cache, logits=logits)


class MockTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        assert [m["role"] for m in messages] == ["system", "user"]
        return messages[1]["content"]

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert not add_special_tokens and return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3 + len(text) % 3]]))

    def encode(self, text, *, add_special_tokens):
        assert text == "\n" and not add_special_tokens
        return [6]

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens
        return " ".join("tok" + str(t) for t in tokens)


def mock_backend():
    g = torch.Generator().manual_seed(20260911)
    caa = {m: {i: torch.randn(8, generator=g) * 0.01 for i in LAYERS} for m in FEATURES}
    hys = {m: {i: torch.randn(1, 2, 2, 2, generator=g) * 0.01 for i in LAYERS} for m in FEATURES}
    for m in FEATURES:
        caa[m].update({i: torch.randn(8, generator=g) * 0.01 for i in range(3, 32, 4)})
    return MockModel(), MockTokenizer(), torch.device("cpu"), {"caa24": caa, "hys": hys}


def smoke_config():
    prompts = read_rows(PACKAGE / "banks/selection8.jsonl")[:2]
    cells = [
        condition("caa24", mode, FEATURES, 0.75, dict.fromkeys(FEATURES, 0.25))
        for mode in ("rss", "raw")
    ]
    cells += [
        condition("hys", mode, FEATURES, 0.75) for mode in ("full", "no_rss", "no_gram", "no_clamp")
    ]
    return (
        {
            "schema_version": 1,
            "job_id": "smoke-cpu",
            "bank": "selection8[:2]",
            "max_new_tokens": 32,
            "norm_mode": "natural",
            "use_windows": False,
            "expected_answers": 12,
            "conditions": cells,
        },
        prompts,
    )
