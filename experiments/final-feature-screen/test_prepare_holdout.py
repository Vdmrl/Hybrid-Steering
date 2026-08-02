from __future__ import annotations

import importlib.util
from pathlib import Path

PATH = Path(__file__).with_name("prepare_holdout.py")
SPEC = importlib.util.spec_from_file_location("prepare_holdout", PATH)
assert SPEC and SPEC.loader
HOLDOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOLDOUT)


def test_select_is_deterministic_and_excludes_list_requests() -> None:
    rows = [
        {
            "instruction": "Explain the role of nutrition in recovery from exercise.",
            "input": "",
        },
        {
            "instruction": "How do tides affect coastal ecosystems over time?",
            "input": "",
        },
        {"instruction": "Describe how recycling affects a city budget.", "input": ""},
        {"instruction": "Give a list of animals in a rainforest.", "input": ""},
    ]
    assert HOLDOUT.select(rows, 3) == HOLDOUT.select(rows, 3)
    assert all("list" not in value.lower() for value in HOLDOUT.select(rows, 3))
