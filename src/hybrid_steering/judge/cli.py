from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .config import load_configs, repo_root
from .runner import judge_task_v3, read_jsonl, run_tasks, task_id, trait_tasks


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind shared LLM-as-a-Judge")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--feature")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--config-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = args.config_root or repo_root()
    features, config = load_configs(root)
    feature_name = args.feature or config.evaluation.default_feature
    if feature_name not in features.features:
        raise SystemExit(
            f"unknown feature {feature_name!r}; choose from {', '.join(features.features)}"
        )
    if "openrouter.ai" in config.base_url and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set")

    feature = features.features[feature_name]
    workers = args.workers or config.generation.workers
    rows = read_jsonl(args.input)
    prompt_name = config.evaluation.prompt
    prompt_path = root / "prompts" / prompt_name
    common: dict[str, Any] = {
        "feature_name": feature_name,
        "feature": feature,
        "template": prompt_path.read_text(encoding="utf-8"),
        "model": config.model,
        "temperature": config.generation.temperature,
        "max_tokens": config.generation.max_output_tokens,
        "retries": config.generation.retries,
        "prompt_name": prompt_name,
        "base_url": config.base_url,
        "timeout": config.generation.timeout_seconds,
        "request_extras": config.generation.request_extras,
    }
    tasks = trait_tasks(rows)

    def worker(task: Any, dry_run: bool = False) -> Any:
        if dry_run:
            row, answer = task
            return task_id(feature_name, row.prompt_id, answer.answer_id)
        return judge_task_v3(task, top_logprobs=config.generation.top_logprobs, **common)

    completed, failed = run_tasks(tasks, worker, output=args.output, workers=workers)
    print(f"complete={completed} failed={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
