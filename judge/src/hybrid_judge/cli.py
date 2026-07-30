from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

from .aggregate import aggregate_pairwise_file
from .config import load_configs
from .provider import openrouter_client
from .runner import (
    compact_trait_task_v4,
    pairwise_task_v2,
    pairwise_tasks,
    read_jsonl,
    run_tasks,
    scalar_task_v2,
    scalar_trait_task_v3,
    scalar_v2_tasks,
)

ROOT = Path(__file__).parents[2]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind shared LLM-as-a-Judge")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode",
        choices=("trait", "trait-audit", "scalar", "pairwise"),
        default="trait",
        help=(
            "trait: compact independent 1-5 score (default); "
            "trait-audit: 1-5 score with exact evidence; "
            "pairwise: optional A/B check; scalar: legacy v2"
        ),
    )
    parser.add_argument("--feature")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--config-root", type=Path, default=ROOT)
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    args = arguments()
    features, config = load_configs(args.config_root)
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
    workers = args.workers or config.generation.workers
    client = openrouter_client(config, api_key)
    rows = read_jsonl(args.input)
    prompt_name = {
        "trait": config.evaluation.trait_prompt,
        "trait-audit": config.evaluation.trait_audit_prompt,
        "scalar": config.evaluation.scalar_prompt,
        "pairwise": config.evaluation.pairwise_prompt,
    }[args.mode]
    prompt_path = args.config_root / "prompts" / prompt_name
    config_path = args.config_root / "config" / "judge.yaml"
    common: dict[str, Any] = {
        "feature_name": feature_name,
        "feature": feature,
        "template": prompt_path.read_text(encoding="utf-8"),
        "client": client,
        "model": config.model,
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

    if args.mode in {"trait", "trait-audit", "scalar"}:
        tasks = scalar_v2_tasks(rows)

        def worker(task: Any, dry_run: bool = False) -> Any:
            row, answer = task
            if dry_run:
                version = {
                    "trait": "trait-compact-v1",
                    "trait-audit": "scalar-v3",
                    "scalar": "scalar-v2",
                }[args.mode]
                return (
                    f"{version}:{features.rubric_version}:{feature_name}:"
                    f"{row.prompt_id}:{answer.answer_id}"
                )
            if args.mode == "trait":
                return compact_trait_task_v4(task, **common)
            if args.mode == "trait-audit":
                return scalar_trait_task_v3(task, **common)
            return scalar_task_v2(task, **common)

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
            return pairwise_task_v2(task, **common)

    completed, failed = run_tasks(
        tasks,
        worker,
        output=args.output,
        workers=workers,
        resume_provenance={
            "judge_model": config.model,
            "provider": config.provider,
            "prompt_version": prompt_name,
            "prompt_sha256": common["prompt_sha256"],
            "rubric_version": features.rubric_version,
            "config_version": config.config_version,
            "config_sha256": common["config_sha256"],
            "seed": args.seed,
            "temperature": config.generation.temperature,
        },
    )
    print(f"complete={completed} failed={failed}", flush=True)
    if args.mode == "pairwise":
        aggregate_path = args.output.with_name(f"{args.output.stem}.aggregated.jsonl")
        count = aggregate_pairwise_file(args.output, aggregate_path)
        print(f"aggregated={count} output={aggregate_path}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
