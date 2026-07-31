from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("five_concept_clamp", PATH)
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
    plans = RUN.condition_plan(toy, toy, 0.5, 0.5)
    assert len(plans) == 26
    assert sum(name.startswith("singleton_") for name in plans) == 5
    assert sum(name.startswith("loo_clamp_") for name in plans) == 5
    assert sum(name.startswith("flip_") for name in plans) == 5


def test_pair_and_triple_count() -> None:
    assert (
        sum(
            1
            for size in (2, 3)
            for _ in __import__("itertools").combinations(RUN.FEATURES, size)
        )
        == 20
    )
