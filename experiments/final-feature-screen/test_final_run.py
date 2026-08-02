from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).with_name("final_run.py")
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("final_run", PATH)
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_condition_sets_have_registered_factorial_sizes() -> None:
    names = ("russian", "optimism", "technical", "numbered_list")
    assert len(RUN.condition_sets(names, "all")) == 15
    assert len(RUN.condition_sets(names + ("persuasive",), "all")) == 31
    assert len(RUN.condition_sets(names, "screen")) == 11
