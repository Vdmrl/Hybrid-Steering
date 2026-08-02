"""Prepare exploratory Judge inputs for the triple smoke."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
BASE = ROOT / "experiments" / "final-feature-screen" / "judge_setup.py"
spec = importlib.util.spec_from_file_location("final_judge_setup", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE}")
setup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = setup
spec.loader.exec_module(setup)

setup.EXPLORATORY["atomic_sentences"] = (
    "atomic sentence structure",
    "compound or multi-clause sentence structure",
    "Atomic structure expresses separate claims as short, complete, standalone sentences.",
    {
        1: "Consistently uses long compound or multi-clause sentences.",
        2: "Mostly uses compound sentences with only incidental short standalone sentences.",
        3: "Mixed, neutral, absent, or too short to judge.",
        4: "Clearly separates most claims into short, complete standalone sentences.",
        5: "Consistently expresses distinct claims as concise, complete standalone sentences.",
    },
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


setup.read_jsonl = read_jsonl


if __name__ == "__main__":
    setup.main()
