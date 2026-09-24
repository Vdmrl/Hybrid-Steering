"""Score a pole-labeled benchmark with the standard Judge v3 task.

Reads the `<stem>.input.jsonl` produced by build_pole_benchmark.py, runs one
Judge call per answer with the configured prompt and rubric, and writes result
rows in the same shape the CLI writes, so pole_metrics.py and metrics.py both
consume them unchanged. Resumes by `task_id`, so an interrupted sweep continues
without re-spending calls.

The provider comes from judge/config/judge.yaml; `--config` selects an
alternative file, which is how a self-hosted OpenAI-compatible endpoint is
addressed without touching the shared default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hybrid_judge.config import load_configs, load_yaml
from hybrid_judge.models import JudgeConfig
from hybrid_judge.provider import judge_client
from hybrid_judge.runner import judge_task_v3, read_jsonl


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="<stem>.input.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--config", type=Path, help="judge.yaml override")
    parser.add_argument("--prompt", help="prompt file name override")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = arguments()
    features, config = load_configs(args.root)
    if args.config:
        config = load_yaml(args.config, JudgeConfig)
    if args.feature not in features.features:
        raise SystemExit(
            f"feature {args.feature!r} is not in concepts/features.yaml; "
            f"known: {', '.join(sorted(features.features))}"
        )

    prompt_name = args.prompt or config.evaluation.prompt
    prompt_path = args.root / "prompts" / prompt_name
    template = prompt_path.read_text(encoding="utf-8")
    config_path = args.config or (args.root / "config" / "judge.yaml")

    rows = read_jsonl(args.inputs)
    if args.limit:
        rows = rows[: args.limit]

    done: set[str] = set()
    if args.output.exists():
        done = {
            json.loads(line)["task_id"]
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        print(f"resuming: {len(done)} rows already scored")

    tasks = [
        (row, answer)
        for row in rows
        for answer in row.answers
        if f"judge-v3:{features.rubric_version}:{args.feature}:"
        f"{row.prompt_id}:{answer.answer_id}" not in done
    ]
    if not tasks:
        print("nothing to do")
        return

    client = judge_client(config, os.environ.get("OPENROUTER_API_KEY", ""))
    workers = args.workers or config.generation.workers
    print(
        f"{len(tasks)} calls | model={config.model} | provider={config.provider} | "
        f"prompt={prompt_name} | feature={args.feature} | workers={workers}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    with args.output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as pool:
        futures = {
            pool.submit(
                judge_task_v3,
                task,
                feature_name=args.feature,
                feature=features.features[args.feature],
                template=template,
                client=client,
                model=config.model,
                provider=config.provider,
                temperature=config.generation.temperature,
                max_tokens=config.generation.max_output_tokens,
                top_logprobs=config.generation.top_logprobs,
                schema_retries=config.generation.schema_retries,
                rubric_version=features.rubric_version,
                prompt_version=prompt_name,
                prompt_sha256=sha256(prompt_path),
                config_version=config.config_version,
                config_sha256=sha256(config_path),
                seed=0,
                request_extras=config.generation.request_extras,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # one bad row must not lose the sweep
                failed += 1
                print(f"  failed: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                continue
            handle.write(result.model_dump_json() + "\n")
            handle.flush()
            written += 1
            if written % 50 == 0:
                print(f"  {written}/{len(tasks)}", flush=True)
    print(f"wrote {written} rows to {args.output} ({failed} failed)")


if __name__ == "__main__":
    main()
