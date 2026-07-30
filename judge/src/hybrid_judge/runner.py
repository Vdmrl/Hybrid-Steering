from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .models import (
    Answer,
    CompactTraitResponseV4,
    FeatureV2,
    JudgeInput,
    PairwiseResponseV2,
    PairwiseResultV2,
    ProvenanceV2,
    ScalarResponseV2,
    ScalarResultV2,
    ScalarTraitResponseV3,
    ScalarTraitResultV3,
    TraitScoreDistribution,
    Usage,
)

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


def render_prompt(template: str, feature: FeatureV2, scale: dict[int, str]) -> str:
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
    extract_json_object: bool = True,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> tuple[str, Usage, str, list[tuple[str, float]] | None]:
    request = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    if logprobs:
        request.update(logprobs=True, top_logprobs=top_logprobs)
    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content
    if not content:
        raise ValueError("judge returned empty content")
    content = content.strip().removeprefix("```json").removesuffix("```").strip()
    token_logprobs = None
    if logprobs:
        positions = getattr(
            getattr(response.choices[0], "logprobs", None), "content", []
        )
        position = next(
            (
                item
                for item in positions or []
                if item.token.strip() in {"1", "2", "3", "4", "5"}
            ),
            None,
        )
        if position is None:
            raise ValueError("judge returned no score-token logprobs")
        token_logprobs = [(item.token, item.logprob) for item in position.top_logprobs]
    if not extract_json_object:
        return content, usage_from(response), response.id, token_logprobs
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge returned no JSON object")
    return (
        content[start : end + 1],
        usage_from(response),
        response.id,
        token_logprobs,
    )


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
    retry_instruction: str | None = None,
    **kwargs: Any,
) -> tuple[ResponseT, list[str], Usage, list[str], list[tuple[str, float]] | None]:
    raw_responses: list[str] = []
    response_ids: list[str] = []
    total_usage = Usage()
    base_system = kwargs["system"]
    for attempt in range(schema_retries + 1):
        if attempt and retry_instruction:
            kwargs["system"] = f"{base_system}\n\n{retry_instruction}"
        content, usage, response_id, token_logprobs = completion(client, **kwargs)
        raw_responses.append(content)
        response_ids.append(response_id)
        total_usage = add_usage(total_usage, usage)
        try:
            parsed = response_model.model_validate_json(content)
            validate(parsed)
            return parsed, raw_responses, total_usage, response_ids, token_logprobs
        except (ValidationError, ValueError):
            continue
    raise SchemaFailure(len(raw_responses), raw_responses)


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
    logprobs: bool = False,
    top_logprobs: int | None = None,
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
        logprobs=logprobs,
        top_logprobs=top_logprobs,
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

    parsed, raw, usage, response_ids, _ = validated_completion(
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


def scalar_trait_task_v3(
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
) -> ScalarTraitResultV3:
    row, answer = task

    def validate(parsed: ScalarTraitResponseV3) -> None:
        if parsed.answer_id != "answer_0":
            raise ValueError("judge returned an unexpected answer_id")
        if parsed.evidence and not parsed.evidence.strip():
            raise ValueError("use an empty string instead of whitespace evidence")
        if parsed.evidence and parsed.evidence not in answer.text:
            raise ValueError("judge evidence is not an exact answer excerpt")
        if parsed.trait_score != 3 and not parsed.evidence:
            raise ValueError("non-neutral trait scores require exact evidence")
        if len(parsed.reason.split()) > 30:
            raise ValueError("judge reason exceeds 30 words")

    parsed, raw, usage, response_ids, _ = validated_completion(
        client,
        response_model=ScalarTraitResponseV3,
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
    return ScalarTraitResultV3(
        task_id=(
            f"scalar-v3:{rubric_version}:{feature_name}:"
            f"{row.prompt_id}:{answer.answer_id}"
        ),
        prompt_id=row.prompt_id,
        answer_id=answer.answer_id,
        feature=feature_name,
        trait_score=parsed.trait_score,
        centered_trait_score=parsed.trait_score - 3,
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


def compact_trait_task_v4(
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
    top_logprobs: int,
    schema_retries: int,
    rubric_version: str,
    prompt_version: str,
    prompt_sha256: str,
    config_version: str,
    config_sha256: str,
    seed: int,
) -> ScalarTraitResultV3:
    row, answer = task
    parsed, raw, usage, response_ids, token_logprobs = validated_completion(
        client,
        response_model=CompactTraitResponseV4,
        validate=lambda _: None,
        schema_retries=schema_retries,
        retry_instruction=(
            "VALIDATION RETRY: Return exactly one ASCII digit from 1 to 5. "
            "Do not return JSON, analysis, or punctuation."
        ),
        model=model,
        system=render_prompt(template, feature, feature.anchors),
        payload={
            "scenario": row.scenario,
            "answer": {"answer_id": "answer_0", "text": answer.text},
        },
        temperature=temperature,
        max_tokens=min(max_tokens, 4),
        extract_json_object=False,
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    score = parsed.root
    distribution = score_distribution(score, token_logprobs or [])
    return ScalarTraitResultV3(
        task_id=(
            f"trait-compact-v1:{rubric_version}:{feature_name}:"
            f"{row.prompt_id}:{answer.answer_id}"
        ),
        prompt_id=row.prompt_id,
        answer_id=answer.answer_id,
        feature=feature_name,
        trait_score=score,
        centered_trait_score=score - 3,
        evidence="",
        reason="",
        score_distribution=distribution,
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
            logprobs=True,
            top_logprobs=top_logprobs,
        ),
    )


def score_distribution(
    chosen_score: int, token_logprobs: list[tuple[str, float]]
) -> TraitScoreDistribution:
    raw = {score: 0.0 for score in range(1, 6)}
    for token, logprob in token_logprobs:
        token = token.strip()
        if token in {"1", "2", "3", "4", "5"}:
            raw[int(token)] += math.exp(logprob)
    valid_mass = sum(raw.values())
    if not valid_mass:
        raise ValueError("judge returned no probabilities for scores 1 through 5")
    probabilities = {score: value / valid_mass for score, value in raw.items()}
    return TraitScoreDistribution(
        probabilities={
            score: round(value, 8) for score, value in probabilities.items()
        },
        expected_score=round(
            sum(score * value for score, value in probabilities.items()), 6
        ),
        chosen_score_probability=round(probabilities[chosen_score], 8),
        entropy=round(
            -sum(value * math.log(value) for value in probabilities.values() if value),
            6,
        ),
        valid_token_mass=round(min(valid_mass, 1.0), 8),
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

    parsed, raw, usage, response_ids, _ = validated_completion(
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
    resume_provenance: dict[str, Any] | None = None,
) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output.exists():
        existing = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if resume_provenance:
        for row in existing:
            provenance = row.get("provenance", {})
            mismatches = [
                key
                for key, expected in resume_provenance.items()
                if provenance.get(key) != expected
            ]
            if mismatches:
                fields = ", ".join(mismatches)
                raise ValueError(
                    f"cannot resume {output}: incompatible provenance fields "
                    f"({fields}); use a new output file"
                )
    done = {row["task_id"] for row in existing}
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
