import importlib
import json
from pathlib import Path

import pytest

from hybrid_steering.layer_selection import select_layer


def test_selects_earliest_near_global_maximum():
    cells = [
        {"layer": 7, "strength": 0.1, "value": 0.2},
        {"layer": 7, "strength": 1, "value": 3},
        {"layer": 31, "strength": 0.1, "value": 2.9},
        {"layer": 31, "strength": 1, "value": 2.95},
    ]
    result = select_layer(cells, [7, 31], [0.1, 1])
    assert result["layer"] == 31 and result["strength"] == 0.1
    assert result["maximum"] == 3  # No assumption that the metric reaches 4.


def test_rate_and_weak_early_plateau():
    cells = [
        {"layer": layer, "strength": s, "value": v}
        for layer, s, v in [(7, 0.1, 0.1), (7, 1, 0.1), (31, 0.1, 0.3), (31, 1, 1)]
    ]
    assert select_layer(cells, [7, 31], [0.1, 1])["layer"] == 31
    assert select_layer([{**c, "value": 0} for c in cells], [7, 31], [0.1, 1])["layer"] is None
    with pytest.raises(ValueError):
        select_layer(cells[:-1], [7, 31], [0.1, 1])


def test_layer_search_writes_selection_and_reuses_existing_pipeline(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    monkeypatch.syspath_prepend(str(directory))
    search = importlib.import_module("residual_layers")
    pareto = importlib.import_module("pareto")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(json.dumps({"key": str(i), "prompt": f"prompt {i}"}) + "\n" for i in range(50))
    )
    suite = {
        "residual_layer_search": {
            "layers": [7, 31],
            "strengths": [0.1, 1],
            "metric_column": "concept_score",
        },
        "screen_dataset": {"path": str(prompts)},
        "benchmarks": [{}],
        "judge_config": "judge.yaml",
    }
    model = {"id": "qwen", "model": "fake", "family": "gdn", "parallel_backend": "single"}
    concept = {"id": "numbered", "feature": "numbered"}
    monkeypatch.setattr(search.pipeline, "load_direction", lambda p: ({7: None, 31: None},))
    monkeypatch.setattr(pareto, "_dataset", lambda *a: {"name": "ifeval"})
    monkeypatch.setattr(search.pipeline, "identity", lambda *a: {"directions": "frozen"})
    monkeypatch.setattr(search.pipeline, "load_plan", lambda *a: ({}, {"judge": []}))
    scored = tmp_path / "scored"
    scored.mkdir()
    values = {(7, 0.1): 0, (7, 1): 3, (31, 0.1): 3, (31, 1): 3}
    records = [
        {
            "condition": f"residual-L{layer}",
            "scale": s,
            "key": str(i),
            "prompt": f"prompt {i}",
            "concept_score": v,
            "evaluable": True,
        }
        for (layer, s), v in values.items()
        for i in range(20)
    ]
    (scored / "judge_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    calls = []

    def run_role(plan, *args):
        data = json.loads(plan.read_text())
        assert len(data["conditions"]) == 2
        assert data["max_new_tokens"] == 512
        calls.append(data)
        return scored, True

    monkeypatch.setattr(pareto, "_run_role", run_role)
    layer = search.search(tmp_path / "suite.json", suite, model, concept, tmp_path, tmp_path, True)
    assert layer == 31 and len(calls) == 1
    folder = tmp_path / "qwen/numbered/residual-layer-search"
    result = json.loads((folder / "selected-residual-layer.json").read_text())
    assert result["metric"] == "concept_score" and result["provenance"] == {"directions": "frozen"}
    assert (folder / "selected-residual-layer.csv").exists()
    assert (folder / "layer-sweep.csv").exists()
    assert (
        search.search(tmp_path / "suite.json", suite, model, concept, tmp_path, tmp_path, True)
        == 31
    )
