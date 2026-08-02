from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PATH = Path(__file__).with_name("summarize.py")
SPEC = importlib.util.spec_from_file_location("final_summary", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def judgment(task_id: str, feature: str, score: int) -> dict:
    return {
        "prompt_id": f"{task_id}:{feature}",
        "feature": feature,
        "trait_score": score,
        "score_distribution": {
            "expected_score": float(score),
            "probabilities": {str(i): float(i == score) for i in range(1, 6)},
        },
        "provenance": {"usage": {"input_tokens": 1, "output_tokens": 1}},
    }


def test_summary_reports_retention_leakage_and_joint_success(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {"features": [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]}
        )
    )
    generations = []
    for prompt in ("p1", "p2"):
        for condition, active in (
            ("baseline", []),
            ("a", ["a"]),
            ("a+b", ["a", "b"]),
            ("a+b+c+d", ["a", "b", "c", "d"]),
        ):
            generations.append(
                {
                    "task_id": f"final:{prompt}:{condition}",
                    "prompt_id": prompt,
                    "condition": condition,
                    "active_features": active,
                    "response": "x",
                }
            )
    path = tmp_path / "generations.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in generations))
    judge = tmp_path / "judge"
    judge.mkdir()
    for feature in ("a", "b", "c", "d", "answer_quality"):
        rows = []
        for generation in generations:
            active = feature in generation["active_features"]
            rows.append(judgment(generation["task_id"], feature, 4 if active else 3))
        (judge / f"{feature}.judgments.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    summary = MODULE.summarize(path, judge, selection)
    assert (
        summary["conditions"]["a"]["delta_vs_baseline"]["a"]["trait_score"]["mean"] == 1
    )
    assert summary["conditions"]["a+b"]["joint_success"]["mean"] == 1
    assert summary["leakage_matrix_expected_score"]["a"]["a"]["mean"] == 1
    assert summary["singleton_retention_expected_score"]["a"]["mean"] == 0
