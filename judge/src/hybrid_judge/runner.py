from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from .models import (
    Answer,
    Feature,
    JudgeInput,
    PairwiseResponse,
    PairwiseResult,
    Provenance,
    ScalarResponse,
    ScalarResult,
    Usage,
)

Mode = Literal["scalar", "pairwise"]


def read_jsonl(path: Path) -> list[JudgeInput]:
    rows = [
        JudgeInput.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row.prompt_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt_id must be unique")
    return rows


def stable_random(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256(":".join([str(seed), *parts]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8]))


def render_prompt(template: str, feature: Feature, scale: dict[int, str]) -> str:
    anchors = "\n".join(f"{score}: {text}" for score, text in sorted(scale.items()))
    exclusions = "\n".join(f"- {item}" for item in feature.exclusions) or "- None"
    return template.format(
        target=feature.target,
        opposite=feature.opposite,
        definition=feature.definition,
        exclusions=exclusions,
        scale=anchors,
    )


def usage_from(response: Any) -> Usage:
    usage = response.usage
    if usage is None:
        return Usage()
    details = getattr(usage, "completion_tokens_details", None)
    return Usage(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
    )


def completion(
    client: OpenAI,
    *,
    model: str,
    system: str,
    payload: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> tuple[str, Usage]:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("judge returned empty content")
    content = content.strip().removeprefix("```json").removesuffix("```").strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge returned no JSON object")
    return content[start : end + 1], usage_from(response)


def scalar_task(
    row: JudgeInput,
    *,
    feature_name: str,
    feature: Feature,
    template: str,
    scale: dict[int, str],
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    rubric_version: str,
    prompt_version: str,
    seed: int,
) -> ScalarResult:
    answers = list(row.answers)
    stable_random(seed, "scalar", feature_name, row.prompt_id).shuffle(answers)
    local_ids = {f"answer_{i}": answer.answer_id for i, answer in enumerate(answers)}
    content, usage = completion(
        client,
        model=model,
        system=render_prompt(template, feature, scale),
        payload={
            "scenario": row.scenario,
            "answers": [
                {"answer_id": local_id, "text": answer.text}
                for local_id, answer in zip(local_ids, answers, strict=True)
            ],
        },
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = ScalarResponse.model_validate_json(content)
    if len(parsed.scores) != len(local_ids) or {
        score.answer_id for score in parsed.scores
    } != set(local_ids):
        raise ValueError("judge returned missing or unexpected answer_id")
    scores = [
        score.model_copy(update={"answer_id": local_ids[score.answer_id]})
        for score in parsed.scores
    ]
    return ScalarResult(
        task_id=f"scalar:{rubric_version}:{feature_name}:{row.prompt_id}",
        prompt_id=row.prompt_id,
        feature=feature_name,
        scores=scores,
        provenance=Provenance(
            judge_model=model,
            prompt_version=prompt_version,
            rubric_version=rubric_version,
            permutation=[answer.answer_id for answer in answers],
            usage=usage,
        ),
    )


def pairwise_tasks(
    rows: Iterable[JudgeInput],
    *,
    feature_name: str,
    both_orders: bool,
    seed: int,
) -> list[tuple[JudgeInput, Answer, Answer, str]]:
    tasks = []
    for row in rows:
        for first, second in itertools.combinations(row.answers, 2):
            pair = [first, second]
            stable_random(
                seed,
                "pairwise",
                feature_name,
                row.prompt_id,
                first.answer_id,
                second.answer_id,
            ).shuffle(pair)
            tasks.append((row, pair[0], pair[1], "one"))
            if both_orders:
                tasks.append((row, pair[1], pair[0], "reverse"))
    return tasks


def pairwise_task(
    task: tuple[JudgeInput, Answer, Answer, str],
    *,
    feature_name: str,
    feature: Feature,
    template: str,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    rubric_version: str,
    prompt_version: str,
) -> PairwiseResult:
    row, left, right, orientation = task
    content, usage = completion(
        client,
        model=model,
        system=render_prompt(template, feature, {}),
        payload={
            "scenario": row.scenario,
            "answer_A": left.text,
            "answer_B": right.text,
        },
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = PairwiseResponse.model_validate_json(content)

    def map_winner(value: str) -> str:
        return {"A": "left", "B": "right", "tie": "tie"}[value]

    pair_key = ":".join(sorted([left.answer_id, right.answer_id]))
    return PairwiseResult(
        task_id=(
            f"pairwise:{rubric_version}:{feature_name}:"
            f"{row.prompt_id}:{pair_key}:{orientation}"
        ),
        prompt_id=row.prompt_id,
        feature=feature_name,
        left_answer_id=left.answer_id,
        right_answer_id=right.answer_id,
        feature_winner=map_winner(parsed.feature_winner),
        quality_winner=map_winner(parsed.quality_winner),
        reason=parsed.reason,
        provenance=Provenance(
            judge_model=model,
            prompt_version=prompt_version,
            rubric_version=rubric_version,
            permutation=[left.answer_id, right.answer_id],
            usage=usage,
        ),
    )


def run_tasks(
    tasks: list[Any],
    worker: Any,
    *,
    output: Path,
    workers: int,
) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {
            json.loads(line)["task_id"]
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    pending = [task for task in tasks if worker(task, dry_run=True) not in done]
    failures = 0
    with (
        output.open("a", encoding="utf-8") as stream,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {pool.submit(worker, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep the queue resumable
                failures += 1
                print(f"judge failed: {exc}", flush=True)
                continue
            stream.write(result.model_dump_json() + "\n")
            stream.flush()
            print(f"judged {index}/{len(pending)}", flush=True)
    return len(pending) - failures, failures
