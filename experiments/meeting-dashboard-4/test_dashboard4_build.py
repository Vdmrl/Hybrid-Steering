import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("build.py")
SPEC = importlib.util.spec_from_file_location("dashboard4_build", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def metric(mean: float, low: float | None = None, high: float | None = None) -> dict:
    return {
        "mean": mean,
        "ci95_low": mean - 0.05 if low is None else low,
        "ci95_high": mean + 0.05 if high is None else high,
    }


def condition(condition_id: str, active: list[str], offset: float = 0.0) -> dict:
    features = {
        feature: {
            "expected": metric(3.4 + offset),
            "delta_expected_vs_baseline": metric(offset),
            "p_ge4": metric(0.5 + offset / 10),
            "delta_p_ge4_vs_baseline": metric(offset / 10),
        }
        for feature in ("joy", "concrete", "optimism", "candor")
    }
    return {
        "condition": condition_id,
        "active_features": active,
        "n": 128,
        "mean_minimum_expected": metric(3.4 + offset),
        "all_active_ge4": metric(0.25 + offset / 10),
        "delta_all_active_ge4_vs_baseline": metric(offset / 10),
        "features": features,
        "quality": metric(4.5 + offset / 10),
        "delta_quality_vs_baseline": metric(offset / 10),
    }


def summary_fixture() -> dict:
    active = ["joy", "concrete", "optimism", "candor"]
    conditions = [condition("baseline", active)]
    conditions.extend(
        condition(method + "_1111", active, index * 0.1)
        for index, method in enumerate(
            ("gdn_raw_r1", "gdn_rss_r1", "gdn_raw_r4", "gdn_rss_r4"), 1
        )
    )
    conditions.extend(
        condition(method + "_0011", ["joy", "concrete"], 0.1)
        for method in ("gdn_raw_r1", "gdn_rss_r1", "gdn_raw_r4", "gdn_rss_r4")
    )
    contrasts = []
    for name in (
        "rss_minus_raw_rank1",
        "rss_minus_raw_rank4",
        "rank4_minus_rank1_raw",
        "rank4_minus_rank1_rss",
    ):
        contrasts.append(
            {
                "contrast": name,
                "mask": "1111",
                "active_features": active,
                "n": 128,
                "delta_mean_minimum_expected": metric(0.1, 0.02, 0.18),
                "delta_all_active_ge4": metric(0.05, -0.01, 0.11),
            }
        )
    contrasts.append(
        {
            "contrast": "rss_minus_raw_rank1",
            "mask": "0011",
            "active_features": ["joy", "concrete"],
            "n": 128,
            "delta_mean_minimum_expected": metric(0.03, -0.04, 0.09),
            "delta_all_active_ge4": metric(0.02, -0.04, 0.08),
        }
    )
    return {
        "selection": {
            "selected_lambda": 0.75,
            "baseline_quality": 4.5,
            "selection_reference": "gdn_raw_r1_1111",
            "candidates": [
                {
                    "lambda": 0.5,
                    "mean_minimum_expected": 3.2,
                    "quality": 4.5,
                    "quality_safe": True,
                },
                {
                    "lambda": 0.75,
                    "mean_minimum_expected": 3.4,
                    "quality": 4.5,
                    "quality_safe": True,
                },
            ],
        },
        "conditions": conditions,
        "contrasts": contrasts,
        "retention": [
            {
                "method": "gdn_raw_r1",
                "feature": "joy",
                "n": 128,
                "full_minus_singleton": metric(-0.1, -0.2, 0.0),
            }
        ],
        "judge_usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "estimated_usd": 0.01,
        },
        "quality_n": 32,
    }


def test_build_describes_exp4_without_results() -> None:
    html = MODULE.build()
    assert "Strong composition" in html
    assert "rank4" in html
    assert "4608" in html
    assert "RSS" in html
    assert "all-four" in html
    assert "__GENERATED__" not in html
    assert "__DATA__" not in html
    assert "summary.json" in html


def test_build_renders_complete_summary_contract() -> None:
    html = MODULE.build(summary_fixture())
    assert "Результаты Exp4" in html
    assert "active_features" in html
    assert "Forest plot" in html
    assert "Singleton retention" in html
    assert "Quality n" in html
    assert "п.п." in html
    assert "балла" in html
    assert "missing" in html
    assert "qualityN=RESULT?.quality_n??null" in html
    assert "qualityNText" in html
    assert "Missing blocks or fields" in html
    assert "expected from pairs" not in html
    assert "__GENERATED__" not in html
    assert "__DATA__" not in html


def test_fixture_is_json_serializable() -> None:
    json.dumps(summary_fixture(), ensure_ascii=False)
