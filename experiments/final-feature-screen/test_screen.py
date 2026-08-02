from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

PATH = Path(__file__).with_name("composition.py")
SPEC = importlib.util.spec_from_file_location("composition", PATH)
assert SPEC and SPEC.loader
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


def test_gram_inverse_recovers_nonorthogonal_coordinates() -> None:
    first = torch.tensor([1.0, 0.0, 0.0])
    second = torch.tensor([1.0, 1.0, 0.0])
    basis = torch.stack([first, second])
    state = 2.0 * first - 0.5 * second
    actual = SCREEN.gram_inverse(basis) @ (basis @ state)
    assert torch.allclose(actual, torch.tensor([2.0, -0.5]), atol=1e-5)
