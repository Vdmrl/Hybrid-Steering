"""Offline checks for the shared direction path and configured evaluation grid."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hybrid_steering.direction import collect_from_pairs
from hybrid_steering.mamba import MambaRunner
from hybrid_steering.runtime import TinyTokenizer


class FakeFalcon(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(model_type="falcon_h1")
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.model = SimpleNamespace(
            layers=[SimpleNamespace(mamba=SimpleNamespace(mamba_rms_norm=False))]
        )
        self.seen = []

    def forward(self, input_ids, past_key_values=None, **_kwargs):
        if past_key_values is None:
            state = torch.zeros(input_ids.shape[0], 1, 1, 1)
            past_key_values = SimpleNamespace(layers=[SimpleNamespace(recurrent_states=[state])])
        state = past_key_values.layers[0].recurrent_states[0]
        self.seen.append(state.clone())
        state.add_(input_ids.sum(-1).view(-1, 1, 1, 1))
        logits = torch.zeros(input_ids.shape[0], 1, 8)
        logits[..., 2] = 1
        return SimpleNamespace(past_key_values=past_key_values, logits=logits)


def test_falcon_direction_and_cache_update():
    model, tokenizer = FakeFalcon(), TinyTokenizer()
    pairs = [("b", "a"), ("d", "c")]
    collected = collect_from_pairs(model, tokenizer, pairs, batch_size=2)
    assert collected.pairs == 2
    assert torch.equal(collected.delta[0], torch.ones(1, 1, 1))
    runner = MambaRunner(model, tokenizer, collected.delta)
    tokens = runner.generate(["ab"], scale=2, max_new_tokens=2)
    assert tokens.shape == (1, 2)
    # The second forward sees the prefill state plus the steered direction.
    expected = float(tokenizer.encode("a")[0] + 2)
    assert torch.equal(model.seen[-2], torch.tensor([[[[expected]]]]))


def test_pipeline_plan_hash_and_method_validation(tmp_path: Path):
    import importlib.util
    import sys

    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("pipeline_run", directory / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"id":"one","prompt":"Hello"}\n', encoding="utf-8")
    plan = {
        "model": "tiny",
        "benchmark": {
            "name": "judge_prompts",
            "path": dataset.name,
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        },
        "max_new_tokens": 4,
        "conditions": [{"name": "base", "method": "baseline", "scales": [0]}],
    }
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(plan), encoding="utf-8")
    loaded, rows = module.load_plan(config)
    assert loaded == plan and rows[0]["_key"] == "one"
    output = tmp_path / "out"
    first = module.generate(config, loaded, rows, output)
    assert len(first) == 1 and first[0]["method"] == "baseline"
    assert module.generate(config, loaded, rows, output) == first
    module.evaluate(config, loaded, rows, output, run_judge=False)
    plan["conditions"][0]["scales"] = [1]
    config.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline"):
        module.load_plan(config)
