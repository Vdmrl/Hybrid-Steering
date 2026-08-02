from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).with_name("report.py")
SPEC = importlib.util.spec_from_file_location("final_report", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_report_uses_score_points_not_percentages() -> None:
    summary = {
        "features": ["a"],
        "selection": {
            "clamp_beta": 1,
            "features": [{"name": "a", "rank": "rank1", "alpha": 2, "c": 1}],
        },
        "conditions": {
            "a": {
                "active_features": ["a"],
                "joint_success": {"mean": 1, "ci95": [1, 1], "n": 2},
                "quality_delta_vs_baseline": {"mean": 0.1, "ci95": [0, 0.2], "n": 2},
                "delta_vs_baseline": {
                    "a": {"expected_score": {"mean": 0.5, "ci95": [0.2, 0.8], "n": 2}}
                },
            }
        },
        "singleton_retention_expected_score": {
            "a": {"mean": -0.1, "ci95": [-0.2, 0], "n": 2}
        },
        "leakage_matrix_expected_score": {
            "a": {"a": {"mean": 0.5, "ci95": [0.2, 0.8], "n": 2}}
        },
        "missing_judge_features": [],
        "judge_usage": {},
    }
    report = MODULE.render(summary)
    assert "+0.500" in report
    assert "percentage" not in report
