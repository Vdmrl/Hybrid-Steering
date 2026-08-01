from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - host-only
    pytest.skip(f"torch runtime is unavailable: {exc}", allow_module_level=True)


MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("style_singleton_run", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feature_contract_and_count() -> None:
    assert len(MODULE.FEATURES) == 6
    assert list(MODULE.FEATURES) == [
        "humorous",
        "adjective_emphasis",
        "action_emphasis",
        "technical",
        "persuasive",
        "narrative",
    ]


def test_self_test() -> None:
    MODULE.self_test()
