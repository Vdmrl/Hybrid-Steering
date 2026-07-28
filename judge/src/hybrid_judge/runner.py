from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .models import (
    Answer,
    Feature,
    FeatureV2,
    JudgeInput,
    PairwiseResponse,
    PairwiseResponseV2,
    PairwiseResult,
    PairwiseResultV2,
    Provenance,
    ProvenanceV2,
    ScalarResponse,
    ScalarResponseV2,
    ScalarResult,
    ScalarResultV2,
    Usage,
)

Mode = Literal["scalar", "pairwise"]
ResponseT = TypeVar("ResponseT", bound=BaseModel)


class SchemaFailure(ValueError):
    def __init__(self, attempts: int, raw_responses: list[str]) -> None:
        super().__init__(f"judge schema validation failed after {attempts} attempts")
        self.raw_responses = raw_responses


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


def render_prompt(
    template: str, feature: Feature | FeatureV2, scale: dict[int, str]
) -> str:
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
) -> tuple[str, Usage, str]:
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
    return content[start : end + 1], usage_from(response), response.id


def add_usage(total: Usage, item: Usage) -> Usage:
    return Usage(
        input_tokens=total.input_tokens + item.input_tokens,
        output_tokens=total.output_tokens + item.output_tokens,
        reasoning_tokens=total.reasoning_tokens + item.reasoning_tokens,
    )


def validated_completion(
    client: OpenAI,
    *,
    response_model: type[ResponseT],
    validate: Callable[[ResponseT], None],
    schema_retries: int,
    **kwargs: Any,
) -> tuple[ResponseT, list[str], Usage, list[str]]:
    raw_responses: list[str] = []
    response_ids: list[str] = []
    total_usage = Usage()
    for _ in range(schema_retries + 1):
        content, usage, response_id = completion(client, **kwargs)
        raw_responses.append(content)
        response_ids.append(response_id)
        total_usage = add_usage(total_usage, usage)
        try:
            parsed = response_model.model_validate_json(content)
            validate(parsed)
            return parsed, raw_responses, total_usage, response_ids
        except (ValidationError, ValueError):
            continue
    raise SchemaFailure(len(raw_responses), raw_responses)


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
    content, usage, _ = completion(
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
    content, usage, _ = completion(
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


def scalar_v2_tasks(rows: Iterable[JudgeInput]) -> list[tuple[JudgeInput, Answer]]:
    return [(row, answer) for row in rows for answer in row.answers]


def provenance_v2(
    *,
    model: str,
    provider: str,
    response_ids: list[str],
    prompt_version: str,
    prompt_sha256: str,
    rubric_version: str,
    config_version: str,
    config_sha256: str,
    answer_order: list[str],
    seed: int,
    temperature: float,
    raw_responses: list[str],
    usage: Usage,
) -> ProvenanceV2:
    return ProvenanceV2(
        judge_model=model,
        provider=provider,
        provider_response_ids=response_ids,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        rubric_version=rubric_version,
        config_version=config_version,
        config_sha256=config_sha256,
        answer_order=answer_order,
        seed=seed,
        temperature=temperature,
        schema_attempts=len(raw_responses),
        timestamp_utc=datetime.now(UTC).isoformat(),
        usage=usage,
        raw_responses=raw_responses,
    )


def scalar_task_v2(
    task: tuple[JudgeInput, Answer],
    *,
    feature_name: str,
    feature: FeatureV2,
    template: str,
    client: OpenAI,
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    schema_retries: int,
    rubric_version: str,
    prompt_version: str,
    prompt_sha256: str,
    config_version: str,
    config_sha256: str,
    seed: int,
) -> ScalarResultV2:
    row, answer = task

    def validate(parsed: ScalarResponseV2) -> None:
        if parsed.answer_id != "answer_0":
            raise ValueError("judge returned an unexpected answer_id")
        if any(not quote.strip() for quote in parsed.evidence):
            raise ValueError("use an empty evidence list instead of empty excerpts")
        if any(quote and quote not in answer.text for quote in parsed.evidence):
            raise ValueError("judge evidence is not an exact answer excerpt")
        if parsed.trait_score != 3 and not parsed.evidence:
            raise ValueError("non-neutral trait scores require exact evidence")
        if len(parsed.reason.split()) > 30:
            raise ValueError("judge reason exceeds 30 words")

    parsed, raw, usage, response_ids = validated_completion(
        client,
        response_model=ScalarResponseV2,
        validate=validate,
        schema_retries=schema_retries,
        model=model,
        system=render_prompt(template, feature, feature.anchors),
        payload={
            "scenario": row.scenario,
            "answer": {"answer_id": "answer_0", "text": answer.text},
        },
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return ScalarResultV2(
        task_id=(
            f"scalar-v2:{rubric_version}:{feature_name}:"
            f"{row.prompt_id}:{answer.answer_id}"
        ),
        prompt_id=row.prompt_id,
        answer_id=answer.answer_id,
        feature=feature_name,
        trait_score=parsed.trait_score,
        centered_trait_score=parsed.trait_score - 3,
        task_fulfillment=parsed.task_fulfillment,
        coherence=parsed.coherence,
        evidence=parsed.evidence,
        reason=parsed.reason,
        provenance=provenance_v2(
            model=model,
            provider=provider,
            response_ids=response_ids,
            prompt_version=prompt_version,
            prompt_sha256=prompt_sha256,
            rubric_version=rubric_version,
            config_version=config_version,
            config_sha256=config_sha256,
            answer_order=[answer.answer_id],
            seed=seed,
            temperature=temperature,
            raw_responses=raw,
            usage=usage,
        ),
    )


def pairwise_task_v2(
    task: tuple[JudgeInput, Answer, Answer, str],
    *,
    feature_name: str,
    feature: FeatureV2,
    template: str,
    client: OpenAI,
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    schema_retries: int,
    rubric_version: str,
    prompt_version: str,
    prompt_sha256: str,
    config_version: str,
    config_sha256: str,
    seed: int,
) -> PairwiseResultV2:
    row, left, right, orientation = task

    def validate(parsed: PairwiseResponseV2) -> None:
        if parsed.evidence_A and parsed.evidence_A not in left.text:
            raise ValueError("evidence_A is not an exact answer excerpt")
        if parsed.evidence_B and parsed.evidence_B not in right.text:
            raise ValueError("evidence_B is not an exact answer excerpt")
        if parsed.trait_winner == "A" and not parsed.evidence_A:
            raise ValueError("trait winner A requires exact evidence_A")
        if parsed.trait_winner == "B" and not parsed.evidence_B:
            raise ValueError("trait winner B requires exact evidence_B")
        if len(parsed.reason.split()) > 30:
            raise ValueError("judge reason exceeds 30 words")

    parsed, raw, usage, response_ids = validated_completion(
        client,
        response_model=PairwiseResponseV2,
        validate=validate,
        schema_retries=schema_retries,
        model=model,
        system=render_prompt(template, feature, feature.anchors),
        payload={
            "scenario": row.scenario,
            "answer_A": left.text,
            "answer_B": right.text,
        },
        temperature=temperature,
        max_tokens=max_tokens,
    )

    def map_winner(value: str) -> str:
        return {"A": "left", "B": "right", "tie": "tie"}[value]

    pair_key = ":".join(sorted([left.answer_id, right.answer_id]))
    return PairwiseResultV2(
        task_id=(
            f"pairwise-v2:{rubric_version}:{feature_name}:"
            f"{row.prompt_id}:{pair_key}:{orientation}"
        ),
        prompt_id=row.prompt_id,
        feature=feature_name,
        orientation=orientation,
        left_answer_id=left.answer_id,
        right_answer_id=right.answer_id,
        trait_winner=map_winner(parsed.trait_winner),
        quality_winner=map_winner(parsed.quality_winner),
        evidence_left=parsed.evidence_A,
        evidence_right=parsed.evidence_B,
        reason=parsed.reason,
        provenance=provenance_v2(
            model=model,
            provider=provider,
            response_ids=response_ids,
            prompt_version=prompt_version,
            prompt_sha256=prompt_sha256,
            rubric_version=rubric_version,
            config_version=config_version,
            config_sha256=config_sha256,
            answer_order=[left.answer_id, right.answer_id],
            seed=seed,
            temperature=temperature,
            raw_responses=raw,
            usage=usage,
        ),
    )


def run_tasks(
    tasks: list[Any],
    worker: Any,
    *,
    output: Path,
    workers: int,
    failures_output: Path | None = None,
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
    failures_output = failures_output or output.with_name(
        f"{output.stem}.failures.jsonl"
    )
    with (
        output.open("a", encoding="utf-8") as stream,
        failures_output.open("a", encoding="utf-8") as failure_stream,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {pool.submit(worker, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep the queue resumable
                failures += 1
                failure = {
                    "task_id": worker(futures[future], dry_run=True),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                raw_responses = getattr(exc, "raw_responses", None)
                if raw_responses is not None:
                    failure["raw_responses"] = raw_responses
                failure_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failure_stream.flush()
                print(f"judge failed: {exc}", flush=True)
                continue
            stream.write(result.model_dump_json() + "\n")
            stream.flush()
            print(f"judged {index}/{len(pending)}", flush=True)
    return len(pending) - failures, failures
