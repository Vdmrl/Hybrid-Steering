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
from hybrid_steering.scoring import wilson


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
        from intervals import bootstrap_ratio
    finally:
        sys.path.pop(0)
    interval = bootstrap_ratio([(1, 1), (0, 3)], draws=1000)
    assert interval == bootstrap_ratio([(1, 1), (0, 3)], draws=1000)
    assert interval["estimate"] == 0.25
    assert interval["low"] <= interval["estimate"] <= interval["high"]
    assert wilson(0, 40)[1] > 0 and wilson(40, 40)[0] < 1
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"id":"one","prompt":"Hello"}\n', encoding="utf-8")
    benchmark = tmp_path / "humaneval.jsonl"
    benchmark.write_text(
        '{"task_id":"t","prompt":"def f():\\n","test":"def check(f):\\n    assert f() == 1\\n","entry_point":"f"}\n',
        encoding="utf-8",
    )
    plan = {
        "model": "tiny",
        "judge_dataset": {
            "path": dataset.name,
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        },
        "bench_dataset": {
            "name": "humaneval",
            "path": benchmark.name,
            "sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        },
        "max_new_tokens": 4,
        "judge": {
            "feature": "fairytale",
            "config": str(directory.parents[1] / "config/judge.yaml"),
        },
        "conditions": [{"name": "base", "method": "baseline", "scales": [0]}],
    }
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(plan), encoding="utf-8")
    loaded, datasets = module.load_plan(config)
    assert loaded == plan and datasets["judge"][0]["_key"] == "one"
    output = tmp_path / "out"
    judge_dir = module.run_directory(config, loaded, datasets["judge"], "judge", output)
    bench_dir = module.run_directory(config, loaded, datasets["benchmark"], "benchmark", output)
    first = module.generate(config, loaded, datasets["judge"], judge_dir, "judge")
    assert len(first) == 1 and first[0]["method"] == "baseline"
    assert module.generate(config, loaded, datasets["judge"], judge_dir, "judge") == first
    assert not module.evaluate(
        config, loaded, datasets["judge"], judge_dir, "judge", run_judge=False
    )

    def fake_score(rows, *_args, **_kwargs):
        for row in rows:
            row.update(concept_score=4, content_quality=5, evaluable=True, hit=1)

    module.score = fake_score
    assert module.evaluate(config, loaded, datasets["judge"], judge_dir, "judge", run_judge=True)
    bench_dir.mkdir(parents=True)
    (bench_dir / "manifest.json").write_text(
        json.dumps(module.identity(config, loaded, datasets["benchmark"], "benchmark"))
    )
    (bench_dir / "benchmark_scores.json").write_text(
        json.dumps(
            {
                "base:0": {
                    "pass_at_1": 0.5,
                    "confidence_intervals": {"pass_at_1": {"low": 0.1, "high": 0.9, "n": 1}},
                }
            }
        )
    )
    report = module.build_report(judge_dir, bench_dir, output)
    assert all(
        (report / name).exists()
        for name in ("rates.csv", "report.json", "comparison.svg", "tradeoff.svg", "report.html")
    )
    chart_row = json.loads((report / "report.json").read_text())["rows"][0]
    assert chart_row["judge_mean"] == 4 and chart_row["judge_delta"] == 0
    import xml.etree.ElementTree as ET

    assert ET.parse(report / "comparison.svg").getroot().tag.endswith("svg")
    assert ET.parse(report / "tradeoff.svg").getroot().tag.endswith("svg")
    # A different benchmark changes only its own cache key; Judge needs no model call.
    benchmark.write_text(benchmark.read_text() + benchmark.read_text().replace('"t"', '"u"'))
    plan["bench_dataset"]["sha256"] = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    config.write_text(json.dumps(plan), encoding="utf-8")
    changed, datasets = module.load_plan(config)
    assert module.run_directory(config, changed, datasets["judge"], "judge", output) == judge_dir
    assert (
        module.run_directory(config, changed, datasets["benchmark"], "benchmark", output)
        != bench_dir
    )
    original_load = module.load_runtime
    module.load_runtime = lambda *_: pytest.fail("Judge answers were regenerated")
    assert module.generate(config, changed, datasets["judge"], judge_dir, "judge") == first
    module.score = lambda *_args, **_kwargs: pytest.fail("Judge scores were recomputed")
    assert module.evaluate(config, changed, datasets["judge"], judge_dir, "judge", run_judge=True)
    module.load_runtime = original_load
    copied = json.loads(json.dumps(plan))
    copied["conditions"].append(
        {"name": "res", "method": "residual_add", "direction": "missing", "scales": [1]}
    )
    config.write_text(json.dumps(copied), encoding="utf-8")
    with pytest.raises(ValueError, match="layer"):
        module.load_plan(config)
    copied["conditions"][-1]["layer"] = 0
    copied["conditions"][-1]["gain"] = -1
    config.write_text(json.dumps(copied), encoding="utf-8")
    with pytest.raises(ValueError, match="gain"):
        module.load_plan(config)
    plan["conditions"][0]["scales"] = [1]
    config.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline"):
        module.load_plan(config)
    plan["conditions"][0]["scales"] = [0, 0]
    config.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        module.load_plan(config)


def test_residual_ignores_causal_attention_mask():
    import importlib.util

    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    spec = importlib.util.spec_from_file_location("pipeline_residual", directory / "residual.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    hidden = torch.zeros(2, 3, 4)
    causal = torch.zeros(2, 1, 3, 8)
    padding = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert module.token_mask(hidden, (), {"attention_mask": causal}) is None
    assert torch.equal(module.token_mask(hidden, (causal, padding), {}), padding)


def test_middle_layers_cover_the_middle_half():
    import importlib.util

    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    spec = importlib.util.spec_from_file_location("pipeline_residual", directory / "residual.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.middle_layers(32) == [8, 12, 16, 20, 24]
    assert module.middle_layers(44) == [11, 16, 21, 26, 31]


def test_summarize_tolerates_lingua_rows_without_quality():
    from hybrid_steering.scoring import summarize

    cells = summarize([{"method": "add", "scale": 1.0, "hit": 1, "repetition": 0.0}])
    assert cells[0]["concept_rate"] == 1
    assert cells[0]["quality"] != cells[0]["quality"]


def test_residual_clamp_replaces_the_projection():
    import importlib.util

    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    spec = importlib.util.spec_from_file_location("pipeline_residual", directory / "residual.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    hidden = torch.tensor([[[3.0, 1.0]]])
    vector = torch.tensor([1.0, 0.0])
    updated = hidden + module.residual_delta(hidden, vector, 4.0, "clamp")
    assert torch.allclose(updated, torch.tensor([[[4.0, 1.0]]]))
    raw = torch.tensor([2.0, 0.0])
    updated = hidden + module.residual_delta(hidden, raw, 4.0, "clamp")
    assert torch.allclose(updated, torch.tensor([[[8.0, 1.0]]]))


def test_residual_add_on_tiny_model():
    import importlib.util

    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    spec = importlib.util.spec_from_file_location("pipeline_residual", directory / "residual.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    from hybrid_steering.runtime import build_tiny

    model, tokenizer = build_tiny()
    runner = module.ResidualRunner(model, tokenizer, torch.ones(model.config.hidden_size), 0)
    tokens = runner.generate(["hello"], scale=1, max_new_tokens=2)
    assert tuple(tokens.shape) == (1, 2)
