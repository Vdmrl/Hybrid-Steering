"""Experiment #4: strong GDN composition with a rank x normalization ablation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).parents[2]
FEATURES = ("joy", "concrete", "optimism", "candor")
RUBRICS = {
    "joy": "joy",
    "concrete": "concrete_language",
    "optimism": "optimism",
    "candor": "principled_candor",
}
STRONG_ALPHAS = {"joy": 4.0, "concrete": 4.0, "optimism": 8.0, "candor": 8.0}
LAMBDAS = (0.5, 0.75, 1.0)
SINGLETONS = (1, 2, 4, 8)
PAIRS = (3, 5, 6, 9, 10, 12)
METADATA = {"task_id", "source_id", "scenario", "_generation"}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


EXP3 = load_module(
    "composition_normalization_v3",
    ROOT / "experiments/composition-normalization-v3/run.py",
)
BASE = EXP3.BASE


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "self-test",
            "smoke",
            "dev",
            "prepare-dev-judge",
            "select",
            "all4",
            "singletons",
            "pairs",
            "prepare-main-judge",
            "summarize",
        ),
    )
    parser.add_argument("--base-directions-dir", type=Path)
    parser.add_argument("--optimism-direction", type=Path)
    parser.add_argument("--joy-direction", type=Path)
    parser.add_argument("--cached-ranks-dir", type=Path)
    parser.add_argument("--dev-prompts", type=Path)
    parser.add_argument("--test-prompts", type=Path)
    parser.add_argument("--baseline-dev-generations", type=Path)
    parser.add_argument("--baseline-main-generations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--quality-test", type=int, default=32)
    return parser.parse_args()


def active(mask: int) -> tuple[str, ...]:
    return tuple(
        feature for index, feature in enumerate(FEATURES) if mask & (1 << index)
    )


def lambda_tag(value: float) -> str:
    return str(value).replace(".", "p")


def scaled_alphas(value: float) -> dict[str, float]:
    return {feature: alpha * value for feature, alpha in STRONG_ALPHAS.items()}


def compose_conditions(
    rank1: dict[str, dict[int, torch.Tensor]],
    rank4: dict[str, dict[int, torch.Tensor]],
    strengths: dict[str, float],
    mask: int,
    *,
    suffix: str = "",
) -> dict[str, list]:
    names = active(mask)
    tail = f"{suffix}_{mask:04b}" if suffix else f"_{mask:04b}"
    return {
        f"gdn_raw_r1{tail}": [(rank1[name], strengths[name], None) for name in names],
        f"gdn_rss_r1{tail}": [(EXP3.rss_direction(rank1, strengths, names), 1.0, None)],
        f"gdn_raw_r4{tail}": [(rank4[name], strengths[name], None) for name in names],
        f"gdn_rss_r4{tail}": [(EXP3.rss_direction(rank4, strengths, names), 1.0, None)],
    }


def directions(args: argparse.Namespace) -> tuple[dict, dict]:
    return EXP3.ranked_directions(args, 1), EXP3.ranked_directions(args, 4)


def generate(
    args: argparse.Namespace,
    rows_path: Path,
    path: Path,
    conditions: dict[str, list],
    prefix: str,
) -> None:
    tokenizer, model = BASE.load_model(args.model)
    BASE.run_records(
        model=model,
        tokenizer=tokenizer,
        rows=EXP3.prompt_rows(rows_path),
        conditions=conditions,
        path=path,
        max_new_tokens=args.max_new_tokens,
        prefix=prefix,
    )


def dev_phase(args: argparse.Namespace) -> None:
    rank1, rank4 = directions(args)
    conditions = {}
    for value in LAMBDAS:
        conditions.update(
            compose_conditions(
                rank1,
                rank4,
                scaled_alphas(value),
                15,
                suffix=f"_l{lambda_tag(value)}",
            )
        )
    generate(
        args,
        args.dev_prompts,
        args.output_dir / "dev-generations.jsonl",
        conditions,
        "exp4:dev",
    )


def smoke_phase(args: argparse.Namespace) -> None:
    rank1, rank4 = directions(args)
    strengths = scaled_alphas(0.5)
    conditions = {}
    for name, value in compose_conditions(rank1, rank4, strengths, 15).items():
        if name in {"gdn_raw_r1_1111", "gdn_rss_r4_1111"}:
            conditions[name] = value
    tokenizer, model = BASE.load_model(args.model)
    row = EXP3.prompt_rows(args.dev_prompts)[0]
    texts, _ = BASE.generate(
        model,
        tokenizer,
        BASE.row_prompt(row),
        conditions,
        24,
    )
    if set(texts) != set(conditions) or any(not text for text in texts.values()):
        raise RuntimeError("smoke generation returned incomplete answers")
    BASE.atomic_json(args.output_dir / "smoke.json", texts)
    print("Experiment #4 GPU smoke passed")


def selected_lambda(args: argparse.Namespace) -> float:
    return float(
        json.loads((args.output_dir / "selection.json").read_text())["selected_lambda"]
    )


def main_phase(args: argparse.Namespace, block: str) -> None:
    rank1, rank4 = directions(args)
    strengths = scaled_alphas(selected_lambda(args))
    conditions = {}
    masks = {"all4": (15,), "singletons": SINGLETONS, "pairs": PAIRS}[block]
    for mask in masks:
        composed = compose_conditions(rank1, rank4, strengths, mask)
        if block == "singletons":
            composed = {
                name: value for name, value in composed.items() if "_raw_" in name
            }
        conditions.update(composed)
    generate(
        args,
        args.test_prompts,
        args.output_dir / f"main-{block}-generations.jsonl",
        conditions,
        f"exp4:{block}",
    )


def merged_generation_rows(args: argparse.Namespace, split: str) -> list[dict]:
    baseline_path = (
        args.baseline_dev_generations
        if split == "dev"
        else args.baseline_main_generations
    )
    baseline = BASE.jsonl(baseline_path)
    merged = {
        row["source_id"]: {
            "source_id": row["source_id"],
            "scenario": row["scenario"],
            "baseline": row["baseline"],
        }
        for row in baseline
    }
    paths = (
        [args.output_dir / "dev-generations.jsonl"]
        if split == "dev"
        else [
            args.output_dir / f"main-{block}-generations.jsonl"
            for block in ("all4", "singletons", "pairs")
        ]
    )
    for path in paths:
        if not path.exists():
            continue
        for row in BASE.jsonl(path):
            target = merged[row["source_id"]]
            target.update(
                {key: value for key, value in row.items() if key not in METADATA}
            )
    return list(merged.values())


def prepare_judge(args: argparse.Namespace, split: str) -> None:
    rows = merged_generation_rows(args, split)
    root = args.output_dir / "judge" / split / "inputs"
    for feature in FEATURES:
        EXP3.write_jsonl(
            root / f"{RUBRICS[feature]}.jsonl",
            [
                {
                    "prompt_id": row["source_id"],
                    "scenario": row["scenario"],
                    "answers": [
                        {"answer_id": key, "text": value}
                        for key, value in row.items()
                        if key not in {"source_id", "scenario"}
                    ],
                    "metadata": {"experiment": "strong-composition-exp4"},
                }
                for row in rows
            ],
        )
    EXP3.write_jsonl(
        root / "quality.jsonl",
        [
            {
                "prompt_id": row["source_id"],
                "scenario": row["scenario"],
                "answers": [
                    {"answer_id": key, "text": value}
                    for key, value in row.items()
                    if key not in {"source_id", "scenario"}
                ],
                "metadata": {"experiment": "strong-composition-exp4"},
            }
            for row in rows[: args.quality_test]
        ],
    )
    print(f"prepared split={split} prompts={len(rows)}")


def result_rows(path: Path) -> list[dict]:
    return BASE.jsonl(path)


def select_phase(args: argparse.Namespace) -> None:
    root = args.output_dir / "judge" / "dev" / "results"
    scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for feature in FEATURES:
        for row in result_rows(root / f"{RUBRICS[feature]}.jsonl"):
            scores[(row["prompt_id"], row["answer_id"])][feature] = row[
                "score_distribution"
            ]["expected_score"]
    quality: dict[str, list[float]] = defaultdict(list)
    for row in result_rows(root / "answer_quality.jsonl"):
        quality[row["answer_id"]].append(row["score_distribution"]["expected_score"])
    baseline_quality = sum(quality["baseline"]) / len(quality["baseline"])
    candidates = []
    for value in LAMBDAS:
        answer_id = f"gdn_raw_r1_l{lambda_tag(value)}_1111"
        minimums = [
            min(values.values())
            for (prompt_id, name), values in scores.items()
            if name == answer_id and set(values) == set(FEATURES)
        ]
        candidate_quality = sum(quality[answer_id]) / len(quality[answer_id])
        candidates.append(
            {
                "lambda": value,
                "mean_minimum_expected": sum(minimums) / len(minimums),
                "quality": candidate_quality,
                "quality_safe": candidate_quality >= baseline_quality - 0.25,
            }
        )
    safe = [row for row in candidates if row["quality_safe"]] or candidates
    chosen = max(safe, key=lambda row: (row["mean_minimum_expected"], -row["lambda"]))
    BASE.atomic_json(
        args.output_dir / "selection.json",
        {
            "selected_lambda": chosen["lambda"],
            "baseline_quality": baseline_quality,
            "selection_reference": "gdn_raw_r1_1111",
            "candidates": candidates,
        },
    )
    print(json.dumps(chosen, indent=2))


def condition_features(answer_id: str) -> tuple[str, ...]:
    return FEATURES if answer_id == "baseline" else active(int(answer_id[-4:], 2))


def bootstrap(values: list[float | bool]) -> dict[str, float]:
    return EXP3.bootstrap_mean([float(value) for value in values])


def summarize_phase(args: argparse.Namespace) -> None:
    root = args.output_dir / "judge" / "main" / "results"
    scores: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    usage = {"input_tokens": 0, "output_tokens": 0}
    for feature in FEATURES:
        for row in result_rows(root / f"{RUBRICS[feature]}.jsonl"):
            scores[(row["prompt_id"], row["answer_id"])][feature] = {
                "hard": row["trait_score"],
                "soft": row["score_distribution"]["expected_score"],
            }
            for key in usage:
                usage[key] += row["provenance"]["usage"][key]
    quality: dict[str, dict[str, float]] = defaultdict(dict)
    for row in result_rows(root / "answer_quality.jsonl"):
        quality[row["answer_id"]][row["prompt_id"]] = row["score_distribution"][
            "expected_score"
        ]
        for key in usage:
            usage[key] += row["provenance"]["usage"][key]

    by_answer: dict[str, dict[str, dict]] = defaultdict(dict)
    for (prompt_id, answer_id), values in scores.items():
        if set(values) == set(FEATURES):
            by_answer[answer_id][prompt_id] = values
    baseline = by_answer["baseline"]

    conditions = []
    for answer_id, rows in sorted(by_answer.items()):
        enabled = condition_features(answer_id)
        prompt_ids = sorted(set(rows) & set(baseline))
        minimums = [
            min(rows[prompt_id][feature]["soft"] for feature in enabled)
            for prompt_id in prompt_ids
        ]
        hard_joint = [
            all(rows[prompt_id][feature]["hard"] >= 4 for feature in enabled)
            for prompt_id in prompt_ids
        ]
        baseline_joint = [
            all(baseline[prompt_id][feature]["hard"] >= 4 for feature in enabled)
            for prompt_id in prompt_ids
        ]
        feature_metrics = {}
        for feature in FEATURES:
            values = [rows[prompt_id][feature]["soft"] for prompt_id in prompt_ids]
            deltas = [
                rows[prompt_id][feature]["soft"] - baseline[prompt_id][feature]["soft"]
                for prompt_id in prompt_ids
            ]
            p4 = [rows[prompt_id][feature]["hard"] >= 4 for prompt_id in prompt_ids]
            base_p4 = [
                baseline[prompt_id][feature]["hard"] >= 4 for prompt_id in prompt_ids
            ]
            feature_metrics[feature] = {
                "expected": bootstrap(values),
                "delta_expected_vs_baseline": bootstrap(deltas),
                "p_ge4": bootstrap(p4),
                "delta_p_ge4_vs_baseline": bootstrap(
                    [left - right for left, right in zip(p4, base_p4, strict=True)]
                ),
            }
        quality_ids = sorted(set(quality[answer_id]) & set(quality["baseline"]))
        conditions.append(
            {
                "condition": answer_id,
                "active_features": list(enabled),
                "n": len(prompt_ids),
                "mean_minimum_expected": bootstrap(minimums),
                "all_active_ge4": bootstrap(hard_joint),
                "delta_all_active_ge4_vs_baseline": bootstrap(
                    [
                        left - right
                        for left, right in zip(hard_joint, baseline_joint, strict=True)
                    ]
                ),
                "features": feature_metrics,
                "quality": (
                    bootstrap(
                        [quality[answer_id][prompt_id] for prompt_id in quality_ids]
                    )
                    if quality_ids
                    else None
                ),
                "delta_quality_vs_baseline": (
                    bootstrap(
                        [
                            quality[answer_id][prompt_id]
                            - quality["baseline"][prompt_id]
                            for prompt_id in quality_ids
                        ]
                    )
                    if quality_ids
                    else None
                ),
            }
        )

    comparisons = (
        ("gdn_rss_r1", "gdn_raw_r1", "rss_minus_raw_rank1"),
        ("gdn_rss_r4", "gdn_raw_r4", "rss_minus_raw_rank4"),
        ("gdn_raw_r4", "gdn_raw_r1", "rank4_minus_rank1_raw"),
        ("gdn_rss_r4", "gdn_rss_r1", "rank4_minus_rank1_rss"),
    )
    contrasts = []
    for mask in (*PAIRS, 15):
        for left, right, name in comparisons:
            left_id, right_id = f"{left}_{mask:04b}", f"{right}_{mask:04b}"
            prompt_ids = sorted(set(by_answer[left_id]) & set(by_answer[right_id]))
            enabled = active(mask)
            minimum_delta = []
            joint_delta = []
            for prompt_id in prompt_ids:
                left_values, right_values = (
                    by_answer[left_id][prompt_id],
                    by_answer[right_id][prompt_id],
                )
                minimum_delta.append(
                    min(left_values[feature]["soft"] for feature in enabled)
                    - min(right_values[feature]["soft"] for feature in enabled)
                )
                joint_delta.append(
                    all(left_values[feature]["hard"] >= 4 for feature in enabled)
                    - all(right_values[feature]["hard"] >= 4 for feature in enabled)
                )
            contrasts.append(
                {
                    "contrast": name,
                    "mask": f"{mask:04b}",
                    "active_features": list(enabled),
                    "n": len(prompt_ids),
                    "delta_mean_minimum_expected": bootstrap(minimum_delta),
                    "delta_all_active_ge4": bootstrap(joint_delta),
                }
            )

    retention = []
    for method in ("gdn_raw_r1", "gdn_rss_r1", "gdn_raw_r4", "gdn_rss_r4"):
        full_id = f"{method}_1111"
        for index, feature in enumerate(FEATURES):
            rank = "r1" if method.endswith("r1") else "r4"
            singleton_id = f"gdn_raw_{rank}_{1 << index:04b}"
            prompt_ids = sorted(set(by_answer[full_id]) & set(by_answer[singleton_id]))
            retention.append(
                {
                    "method": method,
                    "feature": feature,
                    "n": len(prompt_ids),
                    "full_minus_singleton": bootstrap(
                        [
                            by_answer[full_id][prompt_id][feature]["soft"]
                            - by_answer[singleton_id][prompt_id][feature]["soft"]
                            for prompt_id in prompt_ids
                        ]
                    ),
                }
            )

    usage["estimated_usd"] = (
        usage["input_tokens"] * 0.15 + usage["output_tokens"] * 0.60
    ) / 1_000_000
    BASE.atomic_json(
        args.output_dir / "summary.json",
        {
            "selection": json.loads((args.output_dir / "selection.json").read_text()),
            "conditions": conditions,
            "contrasts": contrasts,
            "retention": retention,
            "quality_n": args.quality_test,
            "judge_usage": usage,
        },
    )
    print(f"conditions={len(conditions)} cost=${usage['estimated_usd']:.3f}")


def self_test() -> None:
    assert active(15) == FEATURES
    assert len(PAIRS) == 6
    assert sum(1 for mask in SINGLETONS for _ in ("r1", "r4")) == 8
    toy = {feature: {0: torch.eye(3).reshape(1, 1, 3, 3)} for feature in FEATURES}
    conditions = compose_conditions(toy, toy, STRONG_ALPHAS, 15)
    assert set(conditions) == {
        "gdn_raw_r1_1111",
        "gdn_rss_r1_1111",
        "gdn_raw_r4_1111",
        "gdn_rss_r4_1111",
    }
    print("Experiment #4 self-test passed")


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {
        "self-test": lambda _args: self_test(),
        "smoke": smoke_phase,
        "dev": dev_phase,
        "prepare-dev-judge": lambda value: prepare_judge(value, "dev"),
        "select": select_phase,
        "all4": lambda value: main_phase(value, "all4"),
        "singletons": lambda value: main_phase(value, "singletons"),
        "pairs": lambda value: main_phase(value, "pairs"),
        "prepare-main-judge": lambda value: prepare_judge(value, "main"),
        "summarize": summarize_phase,
    }[args.phase](args)


if __name__ == "__main__":
    main()
