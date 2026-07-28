from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from .aggregate import aggregate_pairwise_file
from .config import load_configs
from .models import FeatureV2, JudgeConfigV2
from .runner import (
    pairwise_task,
    pairwise_task_v2,
    pairwise_tasks,
    read_jsonl,
    run_tasks,
    scalar_task,
    scalar_task_v2,
    scalar_v2_tasks,
)

ROOT = Path(__file__).parents[2]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind shared LLM-as-a-Judge")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--judge-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--mode", choices=("scalar", "pairwise"), default="scalar")
    parser.add_argument("--feature")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--both-orders",
        action="store_true",
        help="Run both answer orders in v1; v2 always requires both orders.",
    )
    parser.add_argument("--config-root", type=Path, default=ROOT)
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    args = arguments()
    features, config = load_configs(args.config_root, args.judge_version)
    feature_name = args.feature or config.evaluation.default_feature
    if feature_name not in features.features:
        raise SystemExit(
            f"unknown feature {feature_name!r}; "
            f"choose from {', '.join(features.features)}"
        )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    feature = features.features[feature_name]
    model = args.model or config.model
    workers = args.workers or config.generation.workers
    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.generation.timeout_seconds,
        max_retries=config.generation.max_retries,
    )
    rows = read_jsonl(args.input)
    prompt_name = (
        config.evaluation.scalar_prompt
        if args.mode == "scalar"
        else config.evaluation.pairwise_prompt
    )
    prompt_path = args.config_root / "prompts" / prompt_name
    template = prompt_path.read_text(encoding="utf-8")

    if args.judge_version == "v1":
        common: dict[str, Any] = {
            "feature_name": feature_name,
            "feature": feature,
            "template": template,
            "client": client,
            "model": model,
            "temperature": config.generation.temperature,
            "max_tokens": config.generation.max_output_tokens,
            "rubric_version": features.rubric_version,
            "prompt_version": prompt_name,
        }
        if args.mode == "scalar":
            tasks: list[Any] = rows

            def worker(task: Any, dry_run: bool = False) -> Any:
                if dry_run:
                    return (
                        f"scalar:{features.rubric_version}:"
                        f"{feature_name}:{task.prompt_id}"
                    )
                return scalar_task(
                    task,
                    scale=config.scale.anchors,
                    seed=args.seed,
                    **common,
                )

        else:
            tasks = pairwise_tasks(
                rows,
                feature_name=feature_name,
                both_orders=args.both_orders,
                seed=args.seed,
            )

            def worker(task: Any, dry_run: bool = False) -> Any:
                row, left, right, orientation = task
                if dry_run:
                    pair_key = ":".join(sorted([left.answer_id, right.answer_id]))
                    return (
                        f"pairwise:{features.rubric_version}:{feature_name}:"
                        f"{row.prompt_id}:{pair_key}:{orientation}"
                    )
                return pairwise_task(task, **common)

    else:
        if not isinstance(feature, FeatureV2) or not isinstance(config, JudgeConfigV2):
            raise TypeError("v2 config was not loaded")
        config_path = args.config_root / "config" / "judge_v2.yaml"
        common_v2: dict[str, Any] = {
            "feature_name": feature_name,
            "feature": feature,
            "template": template,
            "client": client,
            "model": model,
            "provider": config.provider,
            "temperature": config.generation.temperature,
            "max_tokens": config.generation.max_output_tokens,
            "schema_retries": config.generation.schema_retries,
            "rubric_version": features.rubric_version,
            "prompt_version": prompt_name,
            "prompt_sha256": sha256(prompt_path.read_bytes()),
            "config_version": config.config_version,
            "config_sha256": sha256(config_path.read_bytes()),
            "seed": args.seed,
        }
        if args.mode == "scalar":
            tasks = scalar_v2_tasks(rows)

            def worker(task: Any, dry_run: bool = False) -> Any:
                row, answer = task
                if dry_run:
                    return (
                        f"scalar-v2:{features.rubric_version}:{feature_name}:"
                        f"{row.prompt_id}:{answer.answer_id}"
                    )
                return scalar_task_v2(task, **common_v2)

        else:
            tasks = pairwise_tasks(
                rows,
                feature_name=feature_name,
                both_orders=True,
                seed=args.seed,
            )

            def worker(task: Any, dry_run: bool = False) -> Any:
                row, left, right, orientation = task
                if dry_run:
                    pair_key = ":".join(sorted([left.answer_id, right.answer_id]))
                    return (
                        f"pairwise-v2:{features.rubric_version}:{feature_name}:"
                        f"{row.prompt_id}:{pair_key}:{orientation}"
                    )
                return pairwise_task_v2(task, **common_v2)

    completed, failed = run_tasks(
        tasks,
        worker,
        output=args.output,
        workers=workers,
    )
    print(f"complete={completed} failed={failed}", flush=True)
    if args.judge_version == "v2" and args.mode == "pairwise":
        aggregate_path = args.output.with_name(f"{args.output.stem}.aggregated.jsonl")
        count = aggregate_pairwise_file(args.output, aggregate_path)
        print(f"aggregated={count} output={aggregate_path}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
