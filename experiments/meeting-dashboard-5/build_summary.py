"""Build a compact, reproducible summary for the completed Exp5 run.

The runner consumes only the small JSONL Judge outputs and optional generation
text.  It never calls a provider.  Large generation files stay on the server;
the resulting summary is the only data artifact intended for Git.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260801
FEATURES = (
    "french_language",
    "concrete_language",
    "optimism",
    "first_person_voice",
    "bulleted_layout",
)
FEATURE_META = {
    "french_language": {
        "label": "Французский язык",
        "short": "Французский",
        "opposite": "английский / не французский",
    },
    "concrete_language": {
        "label": "Конкретный язык",
        "short": "Конкретность",
        "opposite": "абстрактный язык",
    },
    "optimism": {
        "label": "Оптимизм",
        "short": "Оптимизм",
        "opposite": "пессимизм",
    },
    "first_person_voice": {
        "label": "Первое лицо",
        "short": "Первое лицо",
        "opposite": "третье / безличное лицо",
    },
    "bulleted_layout": {
        "label": "Маркированная структура",
        "short": "Буллеты",
        "opposite": "сплошной текст",
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["prompt_id"]), str(row["answer_id"])


def _expected(row: dict[str, Any]) -> float:
    distribution = row.get("score_distribution") or {}
    if "expected_score" in distribution:
        return float(distribution["expected_score"])
    probabilities = distribution.get("probabilities") or {}
    return float(
        sum(float(score) * float(prob) for score, prob in probabilities.items())
    )


def _metric(values: np.ndarray, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    if values.size == 1:
        number = float(values[0])
        return {"mean": number, "ci95_low": number, "ci95_high": number, "n": 1}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPS, values.size))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "n": int(values.size),
    }


def _metric_matrix(values: np.ndarray, seed: int) -> list[dict[str, float | int]]:
    """Bootstrap columns together, keeping all metrics paired and deterministic."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0:
        return [
            _metric(values[:, column], seed + column)
            for column in range(values.shape[1])
        ]
    if values.shape[0] == 1:
        return [
            {
                "mean": float(values[0, column]),
                "ci95_low": float(values[0, column]),
                "ci95_high": float(values[0, column]),
                "n": 1,
            }
            for column in range(values.shape[1])
        ]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(BOOTSTRAP_REPS, values.shape[0]))
    samples = values[indices].mean(axis=1)
    return [
        {
            "mean": float(values[:, column].mean()),
            "ci95_low": float(np.quantile(samples[:, column], 0.025)),
            "ci95_high": float(np.quantile(samples[:, column], 0.975)),
            "n": int(values.shape[0]),
        }
        for column in range(values.shape[1])
    ]


def _load_feature_rows(
    results_dir: Path,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    rows_by_feature: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for feature in FEATURES:
        rows = _read_jsonl(results_dir / f"{feature}.jsonl")
        if not rows:
            raise ValueError(f"No rows in {feature}.jsonl")
        keyed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            keyed[_key(row)] = {
                "trait_score": float(row["trait_score"]),
                "expected": _expected(row),
                "p_ge4": float(row["trait_score"] >= 4),
                "condition": str(row["answer_id"]),
            }
        rows_by_feature[feature] = keyed
    keys = set(rows_by_feature[FEATURES[0]])
    for feature in FEATURES[1:]:
        if set(rows_by_feature[feature]) != keys:
            raise ValueError(f"Key mismatch in {feature}.jsonl")
    return rows_by_feature


def _load_quality(
    results_dir: Path, keys: list[tuple[str, str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_jsonl(results_dir / "answer_quality.jsonl")
    keyed = {
        _key(row): {
            "trait_score": float(row["trait_score"]),
            "expected": _expected(row),
        }
        for row in rows
    }
    if set(keyed) != set(keys):
        raise ValueError("answer_quality.jsonl does not cover the trait rows")
    return keyed


def _condition_active(condition: str) -> list[str]:
    if condition == "baseline" or condition.startswith(("full_", "flip_")):
        return list(FEATURES)
    if condition.startswith("singleton_"):
        return [condition.removeprefix("singleton_")]
    if condition.startswith(("add_", "clamp_")):
        payload = condition.split("_", 1)[1]
        return [part for part in payload.split("+") if part in FEATURES]
    if condition.startswith(("loo_add_", "loo_clamp_")):
        omitted = condition.split("_", 2)[2]
        return [feature for feature in FEATURES if feature != omitted]
    return list(FEATURES)


def _condition_method(condition: str) -> str:
    if condition == "baseline":
        return "baseline"
    if condition.startswith("singleton_"):
        return "singleton"
    if condition.startswith("full_"):
        return condition
    if condition.startswith("add_"):
        return "add"
    if condition.startswith("clamp_"):
        return "clamp"
    if condition.startswith("loo_add_"):
        return "loo_add"
    if condition.startswith("loo_clamp_"):
        return "loo_clamp"
    if condition.startswith("flip_"):
        return "flip"
    return "other"


def _condition_order(conditions: Iterable[str]) -> list[str]:
    preferred = [
        "baseline",
        *(f"singleton_{feature}" for feature in FEATURES),
        "full_add_raw_r1",
        "full_add_rss_r1",
        "full_add_rss_r4",
        "full_clamp_raw_r1",
        "full_clamp_rss_r1",
        "full_clamp_rss_r4",
    ]
    known = set(conditions)
    ordered = [condition for condition in preferred if condition in known]
    ordered.extend(sorted(known - set(ordered)))
    return ordered


def _stable_seed(text: str, offset: int = 0) -> int:
    """Stable across Python processes (unlike the randomized built-in hash)."""
    return (
        BOOTSTRAP_SEED
        + offset
        + sum((index + 1) * ord(char) for index, char in enumerate(text)) % 10_000
    )


def _provenance(results_dir: Path) -> dict[str, Any]:
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "judgments": 0,
    }
    model = prompt_version = rubric_version = config_version = None
    timestamps: list[str] = []
    for path in [
        results_dir / f"{feature}.jsonl" for feature in (*FEATURES, "answer_quality")
    ]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                provenance = row.get("provenance") or {}
                row_usage = provenance.get("usage") or {}
                usage["input_tokens"] += int(row_usage.get("input_tokens", 0))
                usage["output_tokens"] += int(row_usage.get("output_tokens", 0))
                usage["reasoning_tokens"] += int(row_usage.get("reasoning_tokens", 0))
                usage["judgments"] += 1
                model = model or provenance.get("judge_model")
                prompt_version = prompt_version or provenance.get("prompt_version")
                rubric_version = rubric_version or provenance.get("rubric_version")
                config_version = config_version or provenance.get("config_version")
                if provenance.get("timestamp_utc"):
                    timestamps.append(provenance["timestamp_utc"])
    # OpenRouter's public GPT-4o-mini rates used for the run.  Keep the formula
    # visible so the dashboard is auditable rather than presenting a magic cost.
    cost = (
        usage["input_tokens"] * 0.15 / 1_000_000
        + usage["output_tokens"] * 0.60 / 1_000_000
    )
    return {
        **usage,
        "estimated_usd": cost,
        "input_rate_usd_per_million": 0.15,
        "output_rate_usd_per_million": 0.60,
        "judge_model": model,
        "prompt_version": prompt_version,
        "rubric_version": rubric_version,
        "config_version": config_version,
        "first_timestamp_utc": min(timestamps) if timestamps else None,
        "last_timestamp_utc": max(timestamps) if timestamps else None,
    }


def _deterministic_sanity(generation_dirs: list[Path]) -> dict[str, Any] | None:
    paths: list[Path] = []
    for directory in generation_dirs:
        paths.extend(
            [
                directory / "main-generations.jsonl",
                directory / "extension-generations.jsonl",
            ]
        )
    paths = [path for path in paths if path.exists()]
    if not paths:
        return None
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    by_condition: dict[str, list[str]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(
            str(row.get("response", ""))
        )
    bullet_pattern = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
    first_pattern = re.compile(
        r"\b(?:I|I'm|I've|I'll|me|my|we|our|us|je|j'|me|mon|ma|nous)\b", re.IGNORECASE
    )

    def rate(condition: str, pattern: re.Pattern[str]) -> dict[str, Any] | None:
        values = by_condition.get(condition)
        if not values:
            return None
        hits = np.array([bool(pattern.search(text)) for text in values], dtype=float)
        return {"n": int(hits.size), **_metric(hits, BOOTSTRAP_SEED + len(condition))}

    return {
        "baseline_bullet_rate": rate("baseline", bullet_pattern),
        "singleton_bullet_rate": {
            feature: rate(f"singleton_{feature}", bullet_pattern)
            for feature in FEATURES
        },
        "baseline_first_person_rate": rate("baseline", first_pattern),
        "singleton_first_person_rate": {
            feature: rate(f"singleton_{feature}", first_pattern) for feature in FEATURES
        },
    }


def _selection(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": metric.get("mean"),
        "ci95_low": metric.get("ci95_low"),
        "ci95_high": metric.get("ci95_high"),
    }


def build_summary(
    results_dir: Path,
    generation_dirs: list[Path] | None = None,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    rows_by_feature = _load_feature_rows(results_dir)
    all_keys = sorted(rows_by_feature[FEATURES[0]])
    quality = _load_quality(results_dir, all_keys)

    # Group stable keys by answer condition.  All feature files contain the
    # same key set and answer_id, so this is independent of JSONL ordering.
    condition_keys: dict[str, list[tuple[str, str]]] = {}
    for key in all_keys:
        condition = rows_by_feature[FEATURES[0]][key]["condition"]
        condition_keys.setdefault(condition, []).append(key)
    conditions = _condition_order(condition_keys)
    if not conditions:
        raise ValueError("No conditions found")
    if "baseline" not in condition_keys:
        raise ValueError("The completed run must contain baseline")
    baseline_keys = condition_keys["baseline"]

    def arrays(condition: str, keys: list[tuple[str, str]]) -> tuple[np.ndarray, ...]:
        expected = np.array(
            [
                [rows_by_feature[feature][key]["expected"] for feature in FEATURES]
                for key in keys
            ]
        )
        score = np.array(
            [
                [rows_by_feature[feature][key]["trait_score"] for feature in FEATURES]
                for key in keys
            ]
        )
        p4 = (score >= 4).astype(float)
        quality_values = np.array([quality[key]["expected"] for key in keys])
        active = _condition_active(condition)
        active_indices = [FEATURES.index(feature) for feature in active]
        minimum = expected[:, active_indices].min(axis=1)
        joint = (score[:, active_indices] >= 4).all(axis=1).astype(float)
        return expected, score, p4, quality_values, minimum, joint

    (
        baseline_expected,
        baseline_score,
        baseline_p4,
        baseline_quality,
        baseline_min,
        baseline_joint,
    ) = arrays("baseline", baseline_keys)

    def condition_record(condition: str, keys: list[tuple[str, str]]) -> dict[str, Any]:
        expected, score, p4, quality_values, minimum, joint = arrays(condition, keys)
        # Conditions have one row per prompt and should be paired to baseline.
        baseline_by_prompt = {key[0]: index for index, key in enumerate(baseline_keys)}
        baseline_indices = [baseline_by_prompt[key[0]] for key in keys]
        be = baseline_expected[baseline_indices]
        bs = baseline_score[baseline_indices]
        bp = baseline_p4[baseline_indices]
        bq = baseline_quality[baseline_indices]
        bmin = baseline_min[baseline_indices]
        bjoint = baseline_joint[baseline_indices]
        direct = _metric_matrix(
            np.column_stack([score, expected, p4, minimum, joint, quality_values]),
            _stable_seed(condition),
        )
        deltas = _metric_matrix(
            np.column_stack(
                [
                    score - bs,
                    expected - be,
                    p4 - bp,
                    minimum - bmin,
                    joint - bjoint,
                    quality_values - bq,
                ]
            ),
            _stable_seed(condition, 50_000),
        )
        feature_data: dict[str, Any] = {}
        for index, feature in enumerate(FEATURES):
            feature_data[feature] = {
                "score": _compact_metric(direct[index]),
                "expected": _compact_metric(direct[len(FEATURES) + index]),
                "p_ge4": _compact_metric(direct[2 * len(FEATURES) + index]),
                "delta_score_vs_baseline": _compact_metric(deltas[index]),
                "delta_expected_vs_baseline": _compact_metric(
                    deltas[len(FEATURES) + index]
                ),
                "delta_p_ge4_vs_baseline": _compact_metric(
                    deltas[2 * len(FEATURES) + index]
                ),
            }
        return {
            "condition": condition,
            "method": _condition_method(condition),
            "active_features": _condition_active(condition),
            "n": len(keys),
            "features": feature_data,
            "mean_minimum_expected": _compact_metric(direct[3 * len(FEATURES)]),
            "all_active_ge4": _compact_metric(direct[3 * len(FEATURES) + 1]),
            "delta_mean_minimum_expected_vs_baseline": _compact_metric(
                deltas[3 * len(FEATURES)]
            ),
            "delta_all_active_ge4_vs_baseline": _compact_metric(
                deltas[3 * len(FEATURES) + 1]
            ),
            "quality": _compact_metric(direct[3 * len(FEATURES) + 2]),
            "delta_quality_vs_baseline": _compact_metric(deltas[3 * len(FEATURES) + 2]),
        }

    records = [
        condition_record(condition, condition_keys[condition])
        for condition in conditions
    ]
    record_by_condition = {record["condition"]: record for record in records}

    def contrast(name: str, left: str, right: str) -> dict[str, Any]:
        left_keys = condition_keys[left]
        right_keys = condition_keys[right]
        right_by_prompt = {key[0]: key for key in right_keys}
        pairs = [
            (key, right_by_prompt[key[0]])
            for key in left_keys
            if key[0] in right_by_prompt
        ]
        left_arrays = arrays(left, [pair[0] for pair in pairs])
        right_arrays = arrays(right, [pair[1] for pair in pairs])
        left_min, left_joint, left_quality = (
            left_arrays[4],
            left_arrays[5],
            left_arrays[3],
        )
        right_min, right_joint, right_quality = (
            right_arrays[4],
            right_arrays[5],
            right_arrays[3],
        )
        metric_values = np.column_stack(
            [
                left_min - right_min,
                left_joint - right_joint,
                left_quality - right_quality,
            ]
        )
        metrics = _metric_matrix(metric_values, _stable_seed(name, len(name) * 17))
        return {
            "contrast": name,
            "left": left,
            "right": right,
            "n": int(metric_values.shape[0]),
            "delta_mean_minimum_expected": _compact_metric(metrics[0]),
            "delta_all_active_ge4": _compact_metric(metrics[1]),
            "delta_quality": _compact_metric(metrics[2]),
        }

    contrasts: list[dict[str, Any]] = []
    contrast_specs = [
        ("clamp_minus_add_raw_r1", "full_clamp_raw_r1", "full_add_raw_r1"),
        ("clamp_minus_add_rss_r1", "full_clamp_rss_r1", "full_add_rss_r1"),
        ("clamp_minus_add_rss_r4", "full_clamp_rss_r4", "full_add_rss_r4"),
        ("rss_minus_raw_add_r1", "full_add_rss_r1", "full_add_raw_r1"),
        ("rss_minus_raw_clamp_r1", "full_clamp_rss_r1", "full_clamp_raw_r1"),
        ("rank4_minus_rank1_rss_add", "full_add_rss_r4", "full_add_rss_r1"),
        ("rank4_minus_rank1_rss_clamp", "full_clamp_rss_r4", "full_clamp_rss_r1"),
    ]
    for name, left, right in contrast_specs:
        if left in condition_keys and right in condition_keys:
            contrasts.append(contrast(name, left, right))

    pairs: list[dict[str, Any]] = []
    pair_specs = [
        (f"{first}+{second}", first, second)
        for index, first in enumerate(FEATURES)
        for second in FEATURES[index + 1 :]
    ]
    for label, first, second in pair_specs:
        for method in ("add", "clamp"):
            condition = f"{method}_{first}+{second}"
            if condition not in record_by_condition:
                continue
            record = record_by_condition[condition]
            pairs.append(
                {
                    "pair": label,
                    "features": [first, second],
                    "method": method,
                    "condition": condition,
                    "n": record["n"],
                    "joint": record["all_active_ge4"],
                    "delta_joint_vs_baseline": record[
                        "delta_all_active_ge4_vs_baseline"
                    ],
                    "mean_minimum_expected": record["mean_minimum_expected"],
                    "delta_mean_minimum_expected_vs_baseline": record[
                        "delta_mean_minimum_expected_vs_baseline"
                    ],
                    "quality": record["quality"],
                    "delta_quality_vs_baseline": record["delta_quality_vs_baseline"],
                    "feature_metrics": {
                        feature: record["features"][feature]
                        for feature in (first, second)
                    },
                }
            )
        add = next(
            (item for item in pairs if item["condition"] == f"add_{first}+{second}"),
            None,
        )
        clamp = next(
            (item for item in pairs if item["condition"] == f"clamp_{first}+{second}"),
            None,
        )
        if add and clamp:
            # A separate paired contrast makes the add/clamp comparison visible
            # without pretending that either method is the ground truth.
            pair_contrast = contrast(
                f"clamp_minus_add:{label}", clamp["condition"], add["condition"]
            )
            pair_contrast["pair"] = label
            contrasts.append(pair_contrast)

    retention: list[dict[str, Any]] = []
    # Keep the useful question "does a feature survive composition?" explicit.
    # Exp5 has no single shared full method, so report it for every full method
    # rather than silently selecting one.
    for full_record in [
        record for record in records if record["condition"].startswith("full_")
    ]:
        full_condition = full_record["condition"]
        full_keys = condition_keys[full_condition]
        singleton_by_feature = {
            feature: condition_keys.get(f"singleton_{feature}", [])
            for feature in FEATURES
        }
        for feature in FEATURES:
            singleton_keys = singleton_by_feature[feature]
            singleton_by_prompt = {key[0]: key for key in singleton_keys}
            pair_keys = [
                (key, singleton_by_prompt[key[0]])
                for key in full_keys
                if key[0] in singleton_by_prompt
            ]
            if not pair_keys:
                continue
            full_rows = [pair[0] for pair in pair_keys]
            single_rows = [pair[1] for pair in pair_keys]
            full_values = np.array(
                [rows_by_feature[feature][key]["expected"] for key in full_rows]
            )
            single_values = np.array(
                [rows_by_feature[feature][key]["expected"] for key in single_rows]
            )
            quality_diff = np.array(
                [
                    quality[left]["expected"] - quality[right]["expected"]
                    for left, right in pair_keys
                ]
            )
            retention.append(
                {
                    "method": full_record["method"],
                    "condition": full_condition,
                    "feature": feature,
                    "n": len(pair_keys),
                    "full_minus_singleton": _compact_metric(
                        _metric(
                            full_values - single_values,
                            _stable_seed(f"retention:{full_condition}:{feature}"),
                        )
                    ),
                    "delta_quality": _compact_metric(
                        _metric(
                            quality_diff,
                            _stable_seed(
                                f"retention-quality:{full_condition}:{feature}"
                            ),
                        )
                    ),
                }
            )

    usage = _provenance(results_dir)
    summary = {
        "schema_version": "exp5-dashboard-1",
        "experiment": {
            "id": "five-concept-clamp",
            "title": "Exp5 — five-concept composition and clamp",
            "model": "Qwen3.5-9B",
            "prompts": 128,
            "conditions": len(conditions),
            "generations": len(all_keys),
            "features": list(FEATURES),
            "feature_meta": FEATURE_META,
            "judge_evaluations": usage["judgments"],
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "features": FEATURE_META,
        "conditions": records,
        "singletons": [
            record_by_condition[f"singleton_{feature}"]
            for feature in FEATURES
            if f"singleton_{feature}" in record_by_condition
        ],
        "full_five": [
            record_by_condition[condition]
            for condition in conditions
            if condition.startswith("full_") and condition in record_by_condition
        ],
        "pairs": pairs,
        "loo": [
            record for record in records if record["method"] in {"loo_add", "loo_clamp"}
        ],
        "flip_one": [record for record in records if record["method"] == "flip"],
        "contrasts": contrasts,
        "retention": retention,
        "baseline": record_by_condition.get("baseline"),
        "selection": _selection(selection_path),
        "judge_usage": usage,
        "quality_n": len(quality),
        "trait_n": len(all_keys),
        "deterministic_sanity": _deterministic_sanity(generation_dirs or []),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--generations-dir", type=Path, action="append", default=[])
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.results_dir, args.generations_dir, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "conditions": summary["experiment"]["conditions"],
                "judgments": summary["judge_usage"]["judgments"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
