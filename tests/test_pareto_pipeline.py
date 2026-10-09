"""Offline checks for the thin Pareto selector built on the existing pipeline."""

import importlib
import json
from pathlib import Path

import pytest

from hybrid_steering.scoring import select_pareto_scales


def test_four_targets_use_absolute_error_before_first_peak():
    cells = [
        {"scale": i, "concept_rate": score}
        for i, score in enumerate((0.05, 0.29, 0.49, 0.71, 0.98, 0.4), 1)
    ]
    chosen = select_pareto_scales(cells)
    assert [row["scale"] for row in chosen] == [2, 3, 4, 5]
    assert [row["target"] for row in chosen] == [0.3, 0.5, 0.7, 0.99]
    assert [row["distance"] for row in chosen] == pytest.approx([0.01] * 4)
    assert all(row["scale"] != 6 for row in chosen)
    with pytest.raises(ValueError, match="four distinct"):
        select_pareto_scales(cells[:3])


def test_screen_selects_four_scales_from_fifty_labels(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    monkeypatch.syspath_prepend(str(directory))
    module = importlib.import_module("pareto")
    cases = [
        {"name": "baseline", "method": "baseline", "scales": [0]},
        {"name": "rank1", "method": "gdn_add", "scales": [1, 2, 3, 4, 5]},
    ]
    rows = []
    for scale, count in ((1, 2), (2, 15), (3, 25), (4, 35), (5, 49)):
        for index in range(50):
            rows.append(
                {
                    "condition": "rank1",
                    "scale": scale,
                    "key": str(index),
                    "concept_score": 4 if index < count else 0,
                }
            )
    selected, report = module._select(rows, cases)
    assert selected[0] == cases[0]
    assert selected[1]["scales"] == [2.0, 3.0, 4.0, 5.0]
    assert [row["target"] for row in report["rank1"]] == [0.3, 0.5, 0.7, 0.99]
    rows[0]["concept_score"] = None
    with pytest.raises(ValueError, match="incomplete Judge score"):
        module._select(rows, cases)


def test_frozen_prompt_sets_are_fifty_and_disjoint(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    monkeypatch.syspath_prepend(str(directory))
    module = importlib.import_module("pareto")
    config = directory / "pareto.example.json"
    suite = json.loads(config.read_text())
    screen, confirm = module._prompts(config, suite)
    assert len(module.pipeline.read_lines(Path(screen["path"]))) == 50
    assert len(module.pipeline.read_lines(Path(confirm["path"]))) == 50


def test_screen_rejects_direction_pair_prompts(monkeypatch, tmp_path):
    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    monkeypatch.syspath_prepend(str(directory))
    module = importlib.import_module("pareto")
    config = directory / "pareto.example.json"
    suite = json.loads(config.read_text())
    screen, confirm = module._prompts(config, suite)
    prompt = module.pipeline.read_lines(Path(screen["path"]))[0]["prompt"]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"user_prompt": prompt}) + "\n")
    with pytest.raises(ValueError, match="overlap direction pairs"):
        module._check_pairs(config, {"id": "numbered", "pairs": str(pairs)}, screen, confirm)


def test_condition_grid_reuses_existing_methods(monkeypatch, tmp_path):
    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    monkeypatch.syspath_prepend(str(directory))
    module = importlib.import_module("pareto")
    monkeypatch.setattr(module, "load_direction", lambda _: ({0: object()}, None, None, None))
    monkeypatch.setattr(module, "injected_norm", lambda _direction, rank: 2 if rank else 4)
    for family in ("gdn", "mamba"):
        cases = module._conditions({"family": family}, tmp_path, tmp_path)
        assert sum(len(case["scales"]) for case in cases) == 93
        assert [case["layer"] for case in cases if case["method"].startswith("residual")] == [
            16,
            16,
            20,
            20,
        ]
        assert all(case["scales"] == module.GDN_SCALES for case in cases[1:5])


def test_two_tier_report_keeps_raw_judge_file(monkeypatch, tmp_path):
    directory = Path(__file__).resolve().parents[1] / "experiments/pipeline"
    monkeypatch.syspath_prepend(str(directory))
    report = importlib.import_module("report")
    judge, benchmark = tmp_path / "judge", tmp_path / "benchmark"
    judge.mkdir()
    benchmark.mkdir()
    raw = [
        {
            "condition": "baseline",
            "steering_method": "baseline",
            "scale": 0,
            "key": "p",
            "prompt": "Question",
            "concept_score": 0,
        },
        {
            "condition": "rank1",
            "steering_method": "gdn_add",
            "scale": 1,
            "key": "p",
            "prompt": "Question",
            "concept_score": 3,
        },
    ]
    (judge / "judge_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    (judge / "manifest.json").write_text(json.dumps({"n_tasks": 1}))
    metrics = {
        f"{row['condition']}:{row['scale']}": {
            "prompt_strict": 0.5,
            "confidence_intervals": {"prompt_strict": {"low": 0.1, "high": 0.9, "n": 1}},
        }
        for row in raw
    }
    (benchmark / "benchmark_scores.json").write_text(json.dumps(metrics))
    (benchmark / "manifest.json").write_text("{}")
    result = report.build_report(judge, benchmark, tmp_path, two_tier=True)
    rows = json.loads((result / "report.json").read_text())["rows"]
    assert [row["judge_mean"] for row in rows] == [0, 1]
    assert json.loads((result / "report.json").read_text())["concept_scale"] == (
        "0, 0.5, 1 from Judge 0–4"
    )
    assert (
        json.loads((judge / "judge_scores.jsonl").read_text().splitlines()[1])["concept_score"] == 3
    )
