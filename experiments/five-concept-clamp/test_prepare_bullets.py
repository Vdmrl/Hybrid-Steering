from __future__ import annotations

import importlib.util
from pathlib import Path

PATH = Path(__file__).with_name("prepare_bullets.py")
SPEC = importlib.util.spec_from_file_location("prepare_bullets", PATH)
assert SPEC and SPEC.loader
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_pair_preserves_sentences_and_changes_only_layout() -> None:
    source = "First sentence. Second sentence! Third sentence?"
    bullets, paragraph = PREPARE.paired_layout(source)
    assert bullets == "- First sentence.\n- Second sentence!\n- Third sentence?"
    assert paragraph == source
