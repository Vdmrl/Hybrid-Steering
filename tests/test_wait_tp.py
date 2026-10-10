"""Physical GPU selection for the Theistic TP queue."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "experiments/pipeline/wait_tp.py"
spec = importlib.util.spec_from_file_location("wait_tp", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pair_requires_both_selected_cards_to_be_free():
    cards = {
        1: ("uuid-1", 16376, 169),
        2: ("uuid-2", 16376, 7162),
        3: ("uuid-3", 8192, 0),
    }
    assert module.free_pair(cards, (1, 2)) is None
    cards[2] = ("uuid-2", 16376, 4)
    assert module.free_pair(cards, (1, 2)) == ("uuid-1", "uuid-2")
    with pytest.raises(ValueError, match="two A4000"):
        module.free_pair(cards, (1, 3))
