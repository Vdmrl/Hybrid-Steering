from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("build.py")
SPEC = importlib.util.spec_from_file_location("dashboard5_build", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


FEATURES = [
    "french_language",
    "concrete_language",
    "optimism",
    "first_person_voice",
    "bulleted_layout",
]


def m(value: float, low: float | None = None, high: float | None = None) -> dict:
    return {
        "mean": value,
        "ci95_low": value if low is None else low,
        "ci95_high": value if high is None else high,
    }


def condition(name: str, active: list[str], shift: float = 0.0) -> dict:
    features = {
        feature: {
            "score": m(3.0 + shift),
            "expected": m(3.0 + shift),
            "p_ge4": m(0.4 + shift / 10),
            "delta_score_vs_baseline": m(shift),
            "delta_expected_vs_baseline": m(shift),
            "delta_p_ge4_vs_baseline": m(shift / 10),
        }
        for feature in FEATURES
    }
    return {
        "condition": name,
        "method": "baseline" if name == "baseline" else "singleton",
        "active_features": active,
        "n": 128,
        "features": features,
        "mean_minimum_expected": m(3.0 + shift),
        "all_active_ge4": m(0.2 + shift / 10),
        "delta_mean_minimum_expected_vs_baseline": m(shift),
        "delta_all_active_ge4_vs_baseline": m(shift / 10),
        "quality": m(4.5),
        "delta_quality_vs_baseline": m(0.0),
    }


def minimal_fixture() -> dict:
    baseline = condition("baseline", FEATURES)
    single = condition("singleton_french_language", ["french_language"], 0.5)
    full = condition("full_add_rss_r1", FEATURES, -0.1)
    pair = {
        "pair": "french_language+optimism",
        "features": ["french_language", "optimism"],
        "method": "add",
        "condition": "add_french_language+optimism",
        "n": 128,
        "joint": m(0.4),
        "delta_joint_vs_baseline": m(0.2),
        "mean_minimum_expected": m(2.8),
        "delta_mean_minimum_expected_vs_baseline": m(-0.2),
        "quality": m(4.4),
        "delta_quality_vs_baseline": m(-0.1),
        "feature_metrics": {},
    }
    return {
        "schema_version": "exp5-dashboard-1",
        "experiment": {
            "prompts": 128,
            "conditions": 4,
            "generations": 512,
            "features": FEATURES,
            "bootstrap_reps": 10000,
            "bootstrap_seed": 20260801,
        },
        "features": {
            feature: {"label": feature, "opposite": "opposite"} for feature in FEATURES
        },
        "conditions": [baseline, single, full],
        "baseline": baseline,
        "singletons": [single],
        "full_five": [full],
        "pairs": [pair],
        "loo": [],
        "flip_one": [],
        "contrasts": [
            {
                "contrast": "clamp_minus_add:french_language+optimism",
                "pair": "french_language+optimism",
                "n": 128,
                "delta_mean_minimum_expected": m(0.1),
                "delta_all_active_ge4": m(0.1),
                "delta_quality": m(-0.1),
            }
        ],
        "retention": [
            {
                "method": "add_rss_r1",
                "feature": "french_language",
                "full_minus_singleton": m(-0.1),
                "delta_quality": m(-0.1),
            }
        ],
        "judge_usage": {
            "judgments": 512,
            "input_tokens": 1000,
            "output_tokens": 512,
            "reasoning_tokens": 0,
            "estimated_usd": 0.01,
            "judge_model": "openai/gpt-4o-mini",
            "prompt_version": "judge_v3_compositional.txt",
            "rubric_version": "3.3.0",
            "config_version": "4.2.0",
        },
        "trait_n": 512,
        "quality_n": 128,
        "selection": {"status": "conservative_preselection"},
        "deterministic_sanity": None,
    }


def test_pending_page_has_explicit_waiting_state() -> None:
    html = MODULE.build()
    assert "Эксперимент ещё не завершён" in html
    assert "__DATA__" not in html
    assert "__GENERATED__" not in html


def test_fixture_renders_labels_units_pairs_and_retention() -> None:
    html = MODULE.build(minimal_fixture())
    assert "Французский язык" in html
    assert "п.п." in html
    assert "балла" in html
    assert "french_language+optimism" in html
    assert "Retention" in html
    assert '"trait_n":512' in html
    assert '"quality_n":128' in html
    assert "__DATA__" not in html
    assert "__GENERATED__" not in html


def test_completed_summary_has_all_conditions_and_sane_labels() -> None:
    summary_path = Path(__file__).with_name("exp5_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary["conditions"]) == 47
    assert all(item["n"] == 128 for item in summary["conditions"])
    assert len(summary["pairs"]) == 20
    assert len(summary["retention"]) == 30
    assert summary["judge_usage"]["judgments"] == 36096
    assert summary["features"]["french_language"]["label"] == "Французский язык"
    assert summary["baseline"]["features"]["concrete_language"]["expected"]["mean"] > 4
    assert {item["method"] for item in summary["full_five"]} == {
        "full_add_raw_r1",
        "full_add_rss_r1",
        "full_add_rss_r4",
        "full_clamp_raw_r1",
        "full_clamp_rss_r1",
        "full_clamp_rss_r4",
    }


def test_generated_javascript_is_syntactically_valid(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        return
    html = MODULE.build(minimal_fixture())
    script = html.split("<script>\n", 1)[1].split("\n</script>", 1)[0]
    script_path = tmp_path / "dashboard.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(script_path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
