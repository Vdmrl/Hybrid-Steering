"""Evaluate Judge v3 on manually labeled calibration cases."""

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
from hybrid_judge.runner import judge_task_v3


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("judge_v3_holdout_cases.jsonl"),
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
    done = {row["prompt_id"] for row in existing}
    client = openrouter_client(config, api_key)
    prompt_path = args.root / "prompts" / config.evaluation.prompt
    config_path = args.root / "config" / "judge.yaml"

    def evaluate(case: dict) -> dict:
        answer = Answer(answer_id="candidate", text=case["answer"])
        row = JudgeInput(
            prompt_id=case["prompt_id"],
            scenario=case["scenario"],
            answers=[answer],
        )
        result = judge_task_v3(
            (row, answer),
            feature_name=case["feature"],
            feature=features.features[case["feature"]],
            template=prompt_path.read_text(encoding="utf-8"),
            client=client,
            model=config.model,
            provider=config.provider,
            temperature=config.generation.temperature,
            max_tokens=config.generation.max_output_tokens,
            top_logprobs=config.generation.top_logprobs,
            schema_retries=config.generation.schema_retries,
            rubric_version=features.rubric_version,
            prompt_version=config.evaluation.prompt,
            prompt_sha256=sha256(prompt_path),
            config_version=config.config_version,
            config_sha256=sha256(config_path),
            seed=20260730,
        )
        return {
            "expected_score": case["expected_score"],
            "note": case["note"],
            **result.model_dump(mode="json"),
        }

    pending = [case for case in cases if case["prompt_id"] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("a", encoding="utf-8") as stream,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {pool.submit(evaluate, case): case for case in pending}
        for index, future in enumerate(as_completed(futures), 1):
            case = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve the calibration queue
                failure_path = args.output.with_name(
                    f"{args.output.stem}.failures.jsonl"
                )
                failure = {
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

    rows = [
        json.loads(line)
        for line in args.output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exact = sum(row["trait_score"] == row["expected_score"] for row in rows)
    mae = sum(abs(row["trait_score"] - row["expected_score"]) for row in rows) / len(
        rows
    )
    print(f"judge_v3: exact={exact}/{len(rows)} mae={mae:.3f}")


if __name__ == "__main__":
    main()
