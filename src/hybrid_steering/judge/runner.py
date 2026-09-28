from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .models import (
    Answer,
    Feature,
    JudgeInput,
    JudgeResult,
    ScoreDistribution,
    Usage,
)


class JudgeFailure(ValueError):
    def __init__(self, attempts: int, raw_responses: list[str]) -> None:
        super().__init__(f"judge did not return a score after {attempts} attempts")
        self.raw_responses = raw_responses


def task_id(feature: str, prompt_id: str, answer_id: str) -> str:
    return f"{feature}:{prompt_id}:{answer_id}"


def parse_score(content: str) -> int:
    text = content.strip()
    if text not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"expected a score from 1 to 5, got {content!r}")
    return int(text)


def read_jsonl(path: Path) -> list[JudgeInput]:
    rows = [
        _input(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row.prompt_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt_id must be unique")
    return rows


def _input(raw: dict[str, Any]) -> JudgeInput:
    return JudgeInput(
        prompt_id=raw["prompt_id"],
        scenario=raw["scenario"],
        answers=[Answer(answer_id=item["answer_id"], text=item["text"]) for item in raw["answers"]],
        metadata=dict(raw.get("metadata") or {}),
    )


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


def complete(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[str, Usage, list[tuple[str, float]] | None]:
    """One LiteLLM completion. The key and proxy stay in the environment."""
    import litellm

    proxy = os.environ.get("OPENROUTER_PROXY", "").strip()
    if proxy:
        os.environ.setdefault("HTTPS_PROXY", proxy)
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": os.environ.get("OPENROUTER_API_KEY") or None,
    }
    if base_url:
        request["base_url"] = base_url
    if timeout is not None:
        request["timeout"] = timeout
    if logprobs:
        request["logprobs"] = True
        request["top_logprobs"] = top_logprobs
    if extra:
        request["extra_body"] = dict(extra)
    response = litellm.completion(**request)
    content = response.choices[0].message.content
    if not content:
        raise ValueError("judge returned empty content")
    content = content.strip()
    token_logprobs = None
    if logprobs:
        positions = getattr(getattr(response.choices[0], "logprobs", None), "content", [])
        position = next(
            (item for item in positions or [] if item.token.strip() in {"1", "2", "3", "4", "5"}),
            None,
        )
        if position is None:
            raise ValueError("judge returned no score-token logprobs")
        token_logprobs = [(item.token, item.logprob) for item in position.top_logprobs]
    return content, usage_from(response), token_logprobs


def complete_text(
    prompt: str,
    *,
    model: str,
    base_url: str | None = None,
    max_tokens: int = 8,
    extra: dict[str, Any] | None = None,
) -> str:
    """Score one plain prompt. Used by the 0/1 concept detector."""
    content, _, _ = complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        base_url=base_url,
        extra=extra,
    )
    return content


def add_usage(total: Usage, item: Usage) -> Usage:
    return Usage(
        input_tokens=total.input_tokens + item.input_tokens,
        output_tokens=total.output_tokens + item.output_tokens,
        reasoning_tokens=total.reasoning_tokens + item.reasoning_tokens,
    )


def validated_completion(
    *,
    parse: Callable[[str], Any],
    retries: int,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    extra: dict[str, Any] | None = None,
    retry_instruction: str | None = None,
) -> tuple[Any, list[str], Usage, list[tuple[str, float]] | None]:
    raw_responses: list[str] = []
    total_usage = Usage()
    for attempt in range(retries + 1):
        system_text = system
        if attempt and retry_instruction:
            system_text = f"{system}\n\n{retry_instruction}"
        content, usage, token_logprobs = complete(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            base_url=base_url,
            timeout=timeout,
            extra=extra,
        )
        raw_responses.append(content)
        total_usage = add_usage(total_usage, usage)
        try:
            return parse(content), raw_responses, total_usage, token_logprobs
        except ValueError:
            continue
    raise JudgeFailure(len(raw_responses), raw_responses)


def trait_tasks(rows: Iterable[JudgeInput]) -> list[tuple[JudgeInput, Answer]]:
    return [(row, answer) for row in rows for answer in row.answers]


def judge_task_v3(
    task: tuple[JudgeInput, Answer],
    *,
    feature_name: str,
    feature: Feature,
    template: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_logprobs: int,
    retries: int,
    prompt_name: str,
    base_url: str | None = None,
    timeout: float | None = None,
    request_extras: dict[str, Any] | None = None,
) -> JudgeResult:
    row, answer = task
    score, raw, usage, token_logprobs = validated_completion(
        parse=parse_score,
        retries=retries,
        retry_instruction=(
            "Return exactly one ASCII digit from 1 to 5. "
            "Do not return JSON, analysis, or punctuation."
        ),
        model=model,
        system=render_prompt(template, feature, feature.anchors),
        user=json.dumps(
            {"scenario": row.scenario, "answer": {"answer_id": "answer_0", "text": answer.text}},
            ensure_ascii=False,
        ),
        temperature=temperature,
        max_tokens=min(max_tokens, 4),
        logprobs=True,
        top_logprobs=top_logprobs,
        base_url=base_url,
        timeout=timeout,
        extra=request_extras,
    )
    return JudgeResult(
        task_id=task_id(feature_name, row.prompt_id, answer.answer_id),
        prompt_id=row.prompt_id,
        answer_id=answer.answer_id,
        feature=feature_name,
        trait_score=score,
        centered_trait_score=score - 3,
        score_distribution=score_distribution(score, token_logprobs or []),
        model=model,
        prompt=prompt_name,
        raw=raw[-1],
        usage=usage,
    )


def score_distribution(
    chosen_score: int, token_logprobs: list[tuple[str, float]]
) -> ScoreDistribution:
    raw = {score: 0.0 for score in range(1, 6)}
    for token, logprob in token_logprobs:
        token = token.strip()
        if token in {"1", "2", "3", "4", "5"}:
            raw[int(token)] += math.exp(logprob)
    valid_mass = sum(raw.values())
    if not valid_mass:
        raise ValueError("judge returned no probabilities for scores 1 through 5")
    probabilities = {score: value / valid_mass for score, value in raw.items()}
    return ScoreDistribution(
        probabilities={score: round(value, 8) for score, value in probabilities.items()},
        expected_score=round(sum(score * value for score, value in probabilities.items()), 6),
        chosen_score_probability=round(probabilities[chosen_score], 8),
        entropy=round(
            -sum(value * math.log(value) for value in probabilities.values() if value),
            6,
        ),
        valid_token_mass=round(min(valid_mass, 1.0), 8),
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
    existing = []
    if output.exists():
        existing = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    done = {row["task_id"] for row in existing}
    pending = [task for task in tasks if worker(task, dry_run=True) not in done]
    failures = 0
    failures_output = failures_output or output.with_name(f"{output.stem}.failures.jsonl")
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
            stream.write(result.json() + "\n")
            stream.flush()
            print(f"judged {index}/{len(pending)}", flush=True)
    return len(pending) - failures, failures
