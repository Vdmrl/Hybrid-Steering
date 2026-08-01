from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("urgent_russian_refusal", PATH)
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_gram_coefficients_recovers_nonorthogonal_coordinates() -> None:
    first = torch.tensor([1.0, 0.0, 0.0])
    second = torch.tensor([1.0, 1.0, 0.0])
    state = 2.0 * first - 0.5 * second
    actual = RUN.gram_coefficients(state, [first, second], 1e-8)
    assert torch.allclose(actual, torch.tensor([2.0, -0.5]), atol=1e-5)


def test_primary_plan_has_expected_conditions() -> None:
    toy = {
        name: {0: torch.eye(3).reshape(1, 1, 3, 3) * (index + 1)}
        for index, name in enumerate(RUN.FEATURES)
    }
    plans = RUN.condition_plan(toy)
    assert len(plans) == 16
    assert sum(name.startswith("singleton_") for name in plans) == 4
    assert sum(name.startswith("pair_") for name in plans) == 6
    assert sum(name.startswith("triple_") for name in plans) == 4
    assert sum(name.startswith("full_") for name in plans) == 1


def test_strengths_are_the_validated_singleton_values() -> None:
    assert RUN.DEFAULT_STRENGTHS == {
        "russian_language": 2.0344,
        "optimism": 4.0,
        "casualness": 2.3566,
        "refusal": 2.0,
    }


def test_prepare_judge_groups_conditions() -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_judge", PATH.with_name("prepare_judge.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {
            "source_id": "p1",
            "scenario": "Question",
            "condition": condition,
            "response": condition,
        }
        for condition in ("baseline", "singleton_optimism")
    ]
    grouped = module.grouped_rows(rows)
    assert [answer["answer_id"] for answer in grouped[0]["answers"]] == [
        "baseline",
        "singleton_optimism",
    ]
