from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).with_name("prepare_prompt_split.py")
SPEC = importlib.util.spec_from_file_location("prompt_split", PATH)
assert SPEC and SPEC.loader
SPLIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLIT)


def test_dev_split_is_fixed_and_has_unique_ids(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    SPLIT.write_split(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 16
    assert len({row["id"] for row in rows}) == 16
    assert all(row["prompt"] for row in rows)
