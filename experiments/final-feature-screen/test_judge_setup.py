from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).with_name("judge_setup.py")
SPEC = importlib.util.spec_from_file_location("judge_setup", PATH)
assert SPEC and SPEC.loader
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


def test_blind_inputs_cover_every_feature_and_quality(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        json.dumps(
            {
                "task_id": "final:p0:baseline",
                "scenario": "Explain a decision.",
                "condition": "baseline",
                "response": "A useful answer.",
            }
        )
        + "\n"
    )
    SETUP.write_inputs(tmp_path, generations, ["optimism", "numbered_list"])
    for name in ("optimism", "numbered_list", "answer_quality"):
        row = json.loads((tmp_path / "judge-inputs" / f"{name}.jsonl").read_text())
        assert row["answers"][0]["answer_id"] == "answer_0"
        assert "method" not in row["metadata"]
