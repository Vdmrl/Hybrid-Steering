"""Extend the existing factorial with optimism replacing casualness."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
FEATURES = ("candor", "calm", "concrete", "optimism")
RUBRICS = {
    "candor": "principled_candor",
    "calm": "calm_composure",
    "concrete": "concrete_language",
    "optimism": "optimism",
}
ALPHAS = (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("four_axis_helpers", ROOT / "experiments/four-axis-night/run.py")
FRENCH = load_module(
    "direction_helpers", ROOT / "experiments/calm-french-composition/run.py"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("alpha", "alpha-input", "select", "factorial", "inputs")
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--base-directions-dir", type=Path, required=True)
    parser.add_argument("--base-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--donors", type=int, default=64)
    parser.add_argument("--tuning", type=int, default=40)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def optimism_direction(args: argparse.Namespace) -> dict:
    return FRENCH.load_direction(
        args.output_dir / "directions" / "optimism.safetensors"
    )


def alpha_phase(args: argparse.Namespace) -> None:
    pairs = BASE.jsonl(args.pairs)
    if len(pairs) < args.donors + args.tuning:
        raise RuntimeError("need 64 donor and 40 tuning optimism pairs")
    tokenizer, model = BASE.load_model(args.model)
    direction, _ = FRENCH.ensure_direction(
        model=model,
        tokenizer=tokenizer,
        rows=pairs[: args.donors],
        name="optimism",
        path=args.output_dir / "directions" / "optimism.safetensors",
    )
    BASE.run_records(
        model=model,
        tokenizer=tokenizer,
        rows=pairs[args.donors : args.donors + args.tuning],
        conditions={
            f"a{alpha:g}": [(direction, alpha, None)] for alpha in ALPHAS
        },
        path=args.output_dir / "alpha" / "optimism.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="alpha:optimism",
    )


def alpha_input_phase(args: argparse.Namespace) -> None:
    records = BASE.jsonl(args.output_dir / "alpha" / "optimism.jsonl")
    path = args.output_dir / "judge-inputs" / "alpha-optimism.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt_id": record["task_id"],
                    "scenario": record["scenario"],
                    "answers": [
                        {
                            "answer_id": f"a{alpha:g}",
                            "text": record[f"a{alpha:g}"],
                        }
                        for alpha in ALPHAS
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def select_phase(args: argparse.Namespace) -> None:
    rows = BASE.jsonl(args.output_dir / "judge" / "alpha-optimism.jsonl")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["answer_id"]].append(row)
    expected = {f"a{alpha:g}" for alpha in ALPHAS}
    if set(grouped) != expected:
        raise RuntimeError("incomplete optimism alpha judgments")
    means = {
        name: {
            metric: sum(row[metric] for row in values) / len(values)
            for metric in ("trait_score", "task_fulfillment", "coherence")
        }
        for name, values in grouped.items()
    }
    baseline = means["a0"]
    candidates = []
    for alpha in (1.0, 2.0, 4.0, 8.0):
        item = means[f"a{alpha:g}"]
        candidates.append(
            {
                "alpha": alpha,
                "trait_delta": item["trait_score"] - baseline["trait_score"],
                "quality_drop": max(
                    baseline["task_fulfillment"] - item["task_fulfillment"],
                    baseline["coherence"] - item["coherence"],
                ),
            }
        )
    eligible = [
        item
        for item in candidates
        if item["trait_delta"] >= 0.4 and item["quality_drop"] <= 0.5
    ]
    safe = [item for item in candidates if item["quality_drop"] <= 0.5]
    chosen = (
        min(eligible, key=lambda item: item["alpha"])
        if eligible
        else max(safe, key=lambda item: item["trait_delta"])
        if safe
        else {"alpha": 2.0}
    )
    BASE.atomic_json(
        args.output_dir / "selection.json",
        {
            "optimism": {
                "selected_alpha": chosen["alpha"],
                "means": means,
                "candidates": candidates,
            }
        },
    )


def load_base_directions(args: argparse.Namespace) -> dict[str, dict]:
    return {
        feature: FRENCH.load_direction(
            args.base_directions_dir / f"{feature}.safetensors"
        )
        for feature in FEATURES[:3]
    }


def factorial_phase(args: argparse.Namespace) -> None:
    directions = load_base_directions(args)
    directions["optimism"] = optimism_direction(args)
    base_strengths = json.loads(
        (args.base_output_dir / "selection.json").read_text(encoding="utf-8")
    )
    strengths = {
        feature: float(base_strengths[feature]["selected_alpha"])
        for feature in FEATURES[:3]
    }
    strengths["optimism"] = float(
        json.loads(
            (args.output_dir / "selection.json").read_text(encoding="utf-8")
        )["optimism"]["selected_alpha"]
    )
    conditions = {}
    for low_mask in range(8):
        mask = low_mask | 8
        conditions[f"{mask:04b}"] = [
            (directions[feature], strengths[feature], None)
            for index, feature in enumerate(FEATURES)
            if mask & (1 << index)
        ]
    prompts = BASE.jsonl(args.prompts)[: args.test]
    if len(prompts) != args.test:
        raise RuntimeError("incomplete test prompt split")
    tokenizer, model = BASE.load_model(args.model)
    BASE.run_records(
        model=model,
        tokenizer=tokenizer,
        rows=prompts,
        conditions=conditions,
        path=args.output_dir / "optimism-on.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="optimism-on",
    )


def write_input(
    path: Path,
    records: list[tuple[dict, dict]],
    comparisons: list[tuple[str, str]],
    kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = []
    for old, new in records:
        merged = old | new
        for control, target in comparisons:
            payloads.append(
                {
                    "prompt_id": f"{kind}:{old['source_id']}:{control}",
                    "scenario": old["scenario"],
                    "answers": [
                        {"answer_id": "off", "text": merged[control]},
                        {"answer_id": "on", "text": merged[target]},
                    ],
                    "metadata": {"comparison": control},
                }
            )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payloads),
        encoding="utf-8",
    )


def inputs_phase(args: argparse.Namespace) -> None:
    old = {
        row["source_id"]: row
        for row in BASE.jsonl(args.base_output_dir / "factorial.jsonl")
    }
    new = {
        row["source_id"]: row
        for row in BASE.jsonl(args.output_dir / "optimism-on.jsonl")
    }
    if set(old) != set(new):
        raise RuntimeError("base and optimism prompt IDs differ")
    records = [(old[source_id], new[source_id]) for source_id in sorted(old)]
    jobs = []

    optimism_path = args.output_dir / "judge-inputs" / "main-optimism.jsonl"
    write_input(
        optimism_path,
        records,
        [(f"{mask:04b}", f"{mask | 8:04b}") for mask in range(8)],
        "main",
    )
    jobs.append(
        (
            RUBRICS["optimism"],
            optimism_path,
            args.output_dir / "judge" / "main-optimism.jsonl",
        )
    )

    for index, feature in enumerate(FEATURES[:3]):
        comparisons = []
        for mask in range(8, 16):
            if mask & (1 << index):
                continue
            comparisons.append((f"{mask:04b}", f"{mask | (1 << index):04b}"))
        path = args.output_dir / "judge-inputs" / f"main-{feature}.jsonl"
        write_input(path, records, comparisons, "main")
        jobs.append(
            (
                RUBRICS[feature],
                path,
                args.output_dir / "judge" / f"main-{feature}.jsonl",
            )
        )

    (args.output_dir / "judge-jobs.tsv").write_text(
        "".join(
            f"{rubric}\t{input_path}\t{output_path}\n"
            for rubric, input_path, output_path in jobs
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = arguments()
    if args.self_test:
        assert len(FEATURES) == 4 and FEATURES[-1] == "optimism"
        print("optimism extension self-test passed")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {
        "alpha": alpha_phase,
        "alpha-input": alpha_input_phase,
        "select": select_phase,
        "factorial": factorial_phase,
        "inputs": inputs_phase,
    }[args.phase](args)


if __name__ == "__main__":
    main()
