"""Offline checks; no model downloads, GPU allocation or Judge requests."""

import importlib
import itertools
import json
from pathlib import Path

import pytest

from hybrid_steering.selection import select_spread_levels


@pytest.fixture
def language(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/pipeline"))
    return importlib.import_module("language")


def test_grid_exactly_134_points(language):
    grids = json.loads((language.HERE / "language-grids.json").read_text())
    assert 2 * sum(len(values) for grid in grids.values() for values in grid.values()) == 134
    assert grids["arabic"]["rank1"] == [1.25, 1.4, 1.6, 1.8, 2, 2.25, 2.5, 2.75, 3, 3.5]
    assert grids["french"]["fullrank"] == [0.5, 0.6, 0.75, 0.85, 0.95, 1, 1.2]


def test_maximin_and_no_one_fallback():
    points = [
        {"strength": i + 1, "expression": e} for i, e in enumerate([0.1, 0.2, 0.4, 0.7, 0.9, 0.8])
    ]
    result = select_spread_levels(points)
    assert len(result["selected"]) == 4
    assert result["selected"][-1]["strength"] == 5
    assert not result["observed_expression_one"]
    chosen_gap = result["minimum_expression_distance"]
    possible = [
        min(b["expression"] - a["expression"] for a, b in zip(c, c[1:]))
        for c in itertools.combinations(points[:5], 4)
        if c[-1] == points[4]
    ]
    assert chosen_gap == pytest.approx(max(possible))
    plateau = select_spread_levels([{"strength": i + 1, "expression": 0.5} for i in range(6)])
    assert len(plateau["selected"]) == 4
    assert plateau["used_non_envelope_points"]


def test_null_unevaluable_zero_but_api_failure_refused(language):
    assert language.expression({"concept_score": None, "evaluable": False}) == 0
    assert language.expression({"concept_score": 2}) == 0.5
    assert language.expression({"concept_score": 4}) == 1
    with pytest.raises(ValueError):
        language.expression({"concept_score": None, "evaluable": None})


def test_batch_growth_and_oom_retry_without_missing_rows(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/pipeline"))
    batching = importlib.import_module("batching")
    calls = []

    def generate(batch):
        calls.append(len(batch))
        if len(batch) > 4:
            raise RuntimeError("CUDA out of memory")
        return [[item] for item in batch]

    batches = list(
        batching.adaptive_batches(
            list(range(20)),
            2,
            generate,
            maximum=16,
            target_fraction=0.85,
            memory_fraction=lambda: 0.2,
            reset_memory=lambda: None,
        )
    )
    assert [item for batch, _, _ in batches for item in batch] == list(range(20))
    assert 8 in calls and max(len(batch) for batch, _, _ in batches) == 4
    assert batches[-1][2]["oom_retries"] == 1


def test_screen_selection_before_ifeval_and_export_all_points(language, tmp_path, monkeypatch):
    model = {
        "id": "qwen",
        "model": "Qwen/Qwen3.5-9B",
        "family": "gdn",
        "revision": "pinned",
        "parallel_backend": "single",
    }
    concept = {"id": "french", "feature": "french_language"}
    cases = [
        {"name": "baseline", "method": "baseline", "scales": [0]},
        {"name": "rank1", "method": "gdn_add", "scales": [1, 2, 3, 4, 5]},
    ]
    screen, bench = tmp_path / "screen-answers", tmp_path / "bench-answers"
    screen.mkdir()
    bench.mkdir()
    rated = []
    for case in cases:
        for scale in case["scales"]:
            for key in range(50):
                rated.append(
                    {
                        "condition": case["name"],
                        "steering_method": case["method"],
                        "scale": scale,
                        "key": str(key),
                        "prompt": f"prompt{key}",
                        "response": "text",
                        "concept_score": 4
                        if key < {0: 0, 1: 5, 2: 20, 3: 35, 4: 45, 5: 40}[scale]
                        else 0,
                    }
                )
    (screen / "judge_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rated))
    (screen / "manifest.json").write_text(json.dumps({"n_tasks": 50}))
    monkeypatch.setattr(language.pareto, "_directions", lambda *a: (tmp_path, tmp_path))
    monkeypatch.setattr(language, "conditions", lambda *a: cases)

    def role(path, output, role, judge, model):
        plan = json.loads(path.read_text())
        assert plan["parallel_backend"] == "single"
        assert plan["judge"]["metric"] == "judge_expression"
        if role == "judge":
            return screen, True
        selection = json.loads((tmp_path / "selected-ifeval.json").read_text())
        assert not selection["methods"]["rank1"]["observed_expression_one"]
        assert plan["conditions"][1]["scales"] == [1, 2, 3, 4]
        metrics = {
            f"{c['name']}:{scale}": {
                "prompt_strict": 0.8,
                "confidence_intervals": {"prompt_strict": {"low": 0.76, "high": 0.84, "n": 541}},
            }
            for c in plan["conditions"]
            for scale in c["scales"]
        }
        (bench / "benchmark_scores.json").write_text(json.dumps(metrics))
        (bench / "manifest.json").write_text(json.dumps({"n_tasks": 541}))
        return bench, True

    monkeypatch.setattr(language.pareto, "_run_role", role)
    language.worker(
        {
            "model": model,
            "concept": concept,
            "output": str(tmp_path),
            "config": str(tmp_path / "config.json"),
            "residual_layer": 16,
            "screen": {},
            "benchmark": {"name": "ifeval"},
            "screen_tokens": 512,
            "ifeval_tokens": 1024,
            "initial_batch": 16,
            "auto_batch": {},
            "judge_batch": 8,
            "run_judge": True,
        }
    )
    assert (tmp_path / "COMPLETE.json").exists()
    assert (tmp_path / "pareto.png").exists()
    assert len((tmp_path / "screen.csv").read_text().splitlines()) == 7
    assert len((tmp_path / "measurements.csv").read_text().splitlines()) == 7


def test_single_gpu_subprocess_never_uses_torchrun(language, monkeypatch):
    calls = []
    monkeypatch.setattr(
        language.pareto.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    language.pareto._gpu_call(["worker.py"], {"family": "gdn", "parallel_backend": "single"})
    assert calls[0][0] == [language.sys.executable, "worker.py"]
    assert calls[0][1]["env"]["HYBRID_PARALLEL_BACKEND"] == "single"
