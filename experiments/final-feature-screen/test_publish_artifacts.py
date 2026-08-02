from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PATH = Path(__file__).with_name("publish_artifacts.py")
SPEC = importlib.util.spec_from_file_location("final_publish", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_publish_excludes_absolute_direction_paths(tmp_path: Path) -> None:
    summary = {
        "n_prompts": 128,
        "selection": {
            "clamp_beta": 1,
            "features": [
                {
                    "name": "technical",
                    "direction": "/home/student4/private/technical.safetensors",
                    "rank": "rank4",
                    "alpha": 2,
                    "c": 1,
                    "sign": 1,
                }
            ],
        },
    }
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report_path.write_text("# Report\n", encoding="utf-8")

    MODULE.publish(summary_path, report_path, tmp_path / "published")

    rendered = (tmp_path / "published" / "summary.json").read_text(encoding="utf-8")
    assert "/home/student4" not in rendered
    assert "technical.safetensors" in rendered
    manifest = json.loads(
        (tmp_path / "published" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["prompt_split"]["n_prompts"] == 128
