import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hybrid_judge.aggregate import aggregate_pairwise_v2
from hybrid_judge.config import load_configs
from hybrid_judge.models import PairwiseResultV2, ScalarResponseV2
from hybrid_judge.runner import (
    pairwise_tasks,
    read_jsonl,
    render_prompt,
    run_tasks,
    scalar_v2_tasks,
    validated_completion,
)

ROOT = Path(__file__).parents[1]


def result(
    orientation: str,
    left: str,
    right: str,
    trait_winner: str,
) -> PairwiseResultV2:
    return PairwiseResultV2.model_validate(
        {
            "task_id": f"task-{orientation}",
            "prompt_id": "scenario-001",
            "feature": "optimism",
            "orientation": orientation,
            "left_answer_id": left,
            "right_answer_id": right,
            "trait_winner": trait_winner,
            "quality_winner": "tie",
            "evidence_left": "",
            "evidence_right": "",
            "reason": "clear",
            "provenance": {
                "judge_model": "test",
                "provider": "test",
                "provider_response_ids": ["response"],
                "prompt_version": "pairwise_v2.txt",
                "prompt_sha256": "prompt-hash",
                "rubric_version": "2.0.0",
                "config_version": "2.0.0",
                "config_sha256": "config-hash",
                "answer_order": [left, right],
                "seed": 1,
                "temperature": 0,
                "schema_attempts": 1,
                "timestamp_utc": "2026-07-28T00:00:00+00:00",
                "usage": {},
                "raw_responses": ["{}"],
            },
        }
    )


def test_v2_contracts_and_one_answer_scalar_tasks() -> None:
    features, config = load_configs(ROOT)
    assert set(features.features["french_language"].anchors) == {1, 2, 3, 4, 5}
    feature = features.features["optimism"]
    prompt = render_prompt(
        (ROOT / "prompts" / config.evaluation.scalar_prompt).read_text(
            encoding="utf-8"
        ),
        feature,
        feature.anchors,
    )
    rows = read_jsonl(ROOT / "examples" / "input.example.jsonl")

    assert "1:" in prompt and "5:" in prompt
    assert len(scalar_v2_tasks(rows)) == 2
    assert (
        len(pairwise_tasks(rows, feature_name="optimism", both_orders=True, seed=1))
        == 2
    )
    ScalarResponseV2.model_validate(
        {
            "answer_id": "answer_0",
            "trait_score": 3,
            "task_fulfillment": 5,
            "coherence": 5,
            "evidence": [],
            "reason": "Trait is absent.",
        }
    )
    with pytest.raises(ValidationError):
        ScalarResponseV2.model_validate(
            {
                "answer_id": "answer_0",
                "trait_score": 0,
                "task_fulfillment": 5,
                "coherence": 5,
                "evidence": [],
                "reason": "Invalid trait score.",
            }
        )


def test_pairwise_orders_are_one_experimental_unit() -> None:
    rows = [
        result("one", "baseline", "steered", "right"),
        result("reverse", "steered", "baseline", "left"),
    ]
    aggregate = aggregate_pairwise_v2(rows)[0]
    assert aggregate.orientation_count == 2
    assert aggregate.trait_order_consistent is True
    assert aggregate.trait_winner_answer_id == "steered"


def test_pairwise_order_disagreement_becomes_tie() -> None:
    rows = [
        result("one", "baseline", "steered", "right"),
        result("reverse", "steered", "baseline", "right"),
    ]
    aggregate = aggregate_pairwise_v2(rows)[0]
    assert aggregate.trait_order_consistent is False
    assert aggregate.trait_winner_answer_id == "tie"


def test_schema_failure_is_retried_and_recorded() -> None:
    valid = json.dumps(
        {
            "answer_id": "answer_0",
            "trait_score": 3,
            "task_fulfillment": 5,
            "coherence": 5,
            "evidence": [],
            "reason": "Trait is absent.",
        }
    )
    responses = iter(
        [
            SimpleNamespace(
                id="bad",
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            ),
            SimpleNamespace(
                id="good",
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content=valid))],
            ),
        ]
    )
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return next(responses)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    parsed, raw, _, response_ids = validated_completion(
        client,
        response_model=ScalarResponseV2,
        validate=lambda _: None,
        schema_retries=1,
        model="test",
        system="test",
        payload={},
        temperature=0,
        max_tokens=10,
        reasoning_effort="high",
    )
    assert parsed.trait_score == 3
    assert len(raw) == 2
    assert response_ids == ["bad", "good"]
    assert requests[0]["extra_body"]["reasoning"] == {
        "effort": "high",
        "exclude": True,
    }


def test_queue_persists_failures(tmp_path: Path) -> None:
    output = tmp_path / "scores.jsonl"

    def worker(task: str, dry_run: bool = False) -> str:
        if dry_run:
            return task
        raise ValueError("invalid response")

    assert run_tasks(["task-1"], worker, output=output, workers=1) == (0, 1)
    failure = json.loads(
        (tmp_path / "scores.failures.jsonl").read_text(encoding="utf-8")
    )
    assert failure["task_id"] == "task-1"
