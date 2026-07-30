"""Compare scalar prompt versions on manually labeled hard cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hybrid_judge.config import load_configs
from hybrid_judge.models import Answer, JudgeInput
from hybrid_judge.provider import openrouter_client
from hybrid_judge.runner import scalar_task_v2, scalar_trait_task_v3


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("scalar_hard_cases.jsonl"),
    )
    parser.add_argument(
        "--prompts", nargs="+", default=["scalar_v2.txt", "scalar_v3.txt"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = arguments()
    features, config = load_configs(args.root)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing = []
    if args.output.exists():
        existing = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    done = {(row["prompt_version"], row["prompt_id"]) for row in existing}
    client = openrouter_client(config, api_key)
    config_path = args.root / "config" / "judge.yaml"

    def evaluate(prompt_name: str, case: dict) -> dict:
        prompt_path = args.root / "prompts" / prompt_name
        answer = Answer(answer_id="candidate", text=case["answer"])
        row = JudgeInput(
            prompt_id=case["prompt_id"],
            scenario=case["scenario"],
            answers=[answer],
        )
        if prompt_name == "scalar_v3.txt":
            result = scalar_trait_task_v3(
                (row, answer),
                feature_name=case["feature"],
                feature=features.features[case["feature"]],
                template=prompt_path.read_text(encoding="utf-8"),
                client=client,
                model=config.model,
                provider=config.provider,
                temperature=config.generation.temperature,
                max_tokens=config.generation.max_output_tokens,
                schema_retries=config.generation.schema_retries,
                rubric_version=features.rubric_version,
                prompt_version=prompt_name,
                prompt_sha256=sha256(prompt_path),
                config_version=config.config_version,
                config_sha256=sha256(config_path),
                seed=20260729,
            )
            result_data = result.model_dump(mode="json")
        else:
            result = scalar_task_v2(
                (row, answer),
                feature_name=case["feature"],
                feature=features.features[case["feature"]],
                template=prompt_path.read_text(encoding="utf-8"),
                client=client,
                model=config.model,
                provider=config.provider,
                temperature=config.generation.temperature,
                max_tokens=config.generation.max_output_tokens,
                schema_retries=config.generation.schema_retries,
                rubric_version=features.rubric_version,
                prompt_version=prompt_name,
                prompt_sha256=sha256(prompt_path),
                config_version=config.config_version,
                config_sha256=sha256(config_path),
                seed=20260729,
            )
            result_data = result.model_dump(mode="json")
        return {
            "prompt_version": prompt_name,
            "prompt_id": case["prompt_id"],
            "feature": case["feature"],
            "expected_score": case["expected_score"],
            "note": case["note"],
            **result_data,
        }

    pending = [
        (prompt_name, case)
        for prompt_name in args.prompts
        for case in cases
        if (prompt_name, case["prompt_id"]) not in done
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("a", encoding="utf-8") as stream,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {
            pool.submit(evaluate, prompt_name, case): (prompt_name, case)
            for prompt_name, case in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            prompt_name, case = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve the calibration queue
                failure_path = args.output.with_name(
                    f"{args.output.stem}.failures.jsonl"
                )
                failure = {
                    "prompt_version": prompt_name,
                    "prompt_id": case["prompt_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_responses": getattr(exc, "raw_responses", []),
                }
                with failure_path.open("a", encoding="utf-8") as failure_stream:
                    failure_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(f"failed {index}/{len(pending)}: {case['prompt_id']}")
                continue
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"evaluated {index}/{len(pending)}", flush=True)

    rows = (
        existing
        + [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][len(existing) :]
    )
    for prompt_name in args.prompts:
        prompt_rows = [row for row in rows if row["prompt_version"] == prompt_name]
        exact = sum(row["trait_score"] == row["expected_score"] for row in prompt_rows)
        mae = sum(
            abs(row["trait_score"] - row["expected_score"]) for row in prompt_rows
        ) / len(prompt_rows)
        print(f"{prompt_name}: exact={exact}/{len(prompt_rows)} mae={mae:.3f}")


if __name__ == "__main__":
    main()
