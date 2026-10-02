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
    assert all((report / name).exists() for name in ("rates.csv", "report.json", "comparison.svg"))
    chart_row = json.loads((report / "report.json").read_text())["rows"][0]
    assert chart_row["judge_mean"] == 4 and chart_row["judge_delta"] == 0
    import xml.etree.ElementTree as ET

    assert ET.parse(report / "comparison.svg").getroot().tag.endswith("svg")
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
    plan["conditions"][0]["scales"] = [1]
    config.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline"):
        module.load_plan(config)
    plan["conditions"][0]["scales"] = [0, 0]
    config.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        module.load_plan(config)
