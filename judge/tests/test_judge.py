import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_judge.aggregate import aggregate_pairwise_v2
from hybrid_judge.cli import arguments
from hybrid_judge.config import load_configs
from hybrid_judge.models import (
    Answer,
    FeatureV2,
    JudgeInput,
    PairwiseResultV2,
    ScalarTraitResponseV3,
)
from hybrid_judge.runner import (
    SchemaFailure,
    compact_trait_task_v4,
    pairwise_tasks,
    read_jsonl,
    render_prompt,
    run_tasks,
    scalar_trait_task_v3,
    trait_tasks,
    validated_completion,
)

ROOT = Path(__file__).parents[1]


def test_cli_defaults_to_v3_and_rejects_legacy_scalar(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hybrid-judge", "input.jsonl", "output.jsonl"])
    assert arguments().mode == "trait"

    monkeypatch.setattr(
        sys,
        "argv",
        ["hybrid-judge", "input.jsonl", "output.jsonl", "--mode", "scalar"],
    )
    with pytest.raises(SystemExit):
        arguments()


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


def test_contracts_and_one_answer_trait_tasks() -> None:
    features, config = load_configs(ROOT)
    assert set(features.features["french_language"].anchors) == {1, 2, 3, 4, 5}
    feature = features.features["optimism"]
    prompt = render_prompt(
        (ROOT / "prompts" / config.evaluation.trait_prompt).read_text(encoding="utf-8"),
        feature,
        feature.anchors,
    )
    rows = read_jsonl(ROOT / "examples" / "input.example.jsonl")

    assert "1:" in prompt and "5:" in prompt
    assert len(trait_tasks(rows)) == 2
    assert (
        len(pairwise_tasks(rows, feature_name="optimism", both_orders=True, seed=1))
        == 2
    )


def test_v3_trait_only_contract() -> None:
    _, config = load_configs(ROOT)
    parsed = ScalarTraitResponseV3.model_validate(
        {
            "answer_id": "answer_0",
            "trait_score": 3,
            "evidence": "",
            "reason": "The trait is absent.",
        }
    )
    assert parsed.trait_score == 3
    assert "task_fulfillment" not in parsed.model_fields_set
    assert config.evaluation.trait_prompt == "trait_compact_v1.txt"
    assert config.evaluation.trait_audit_prompt == "scalar_v3.txt"


def test_compact_trait_accepts_one_digit_and_caps_output_tokens() -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        probabilities = {1: 0.02, 2: 0.03, 3: 0.20, 4: 0.60, 5: 0.15}
        top_logprobs = [
            SimpleNamespace(token=str(score), logprob=math.log(probability))
            for score, probability in probabilities.items()
        ]
        return SimpleNamespace(
            id="response",
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="4"),
                    logprobs=SimpleNamespace(
                        content=[SimpleNamespace(token="4", top_logprobs=top_logprobs)]
                    ),
                )
            ],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    answer = Answer(answer_id="candidate", text="A clearly targeted answer.")
    row = JudgeInput(prompt_id="scenario", scenario="Scenario", answers=[answer])
    feature = FeatureV2(
        target="target",
        opposite="opposite",
        definition="definition",
        anchors={score: str(score) for score in range(1, 6)},
    )

    result = compact_trait_task_v4(
        (row, answer),
        feature_name="feature",
        feature=feature,
        template="{target}\n{opposite}\n{definition}\n{exclusions}\n{scale}",
        client=client,
        model="test",
        provider="test",
        temperature=0,
        max_tokens=256,
        top_logprobs=20,
        schema_retries=0,
        rubric_version="1",
        prompt_version="trait_compact_v1.txt",
        prompt_sha256="prompt-hash",
        config_version="1",
        config_sha256="config-hash",
        seed=1,
    )

    assert result.trait_score == 4
    assert result.centered_trait_score == 1
    assert result.score_distribution is not None
    assert result.score_distribution.probabilities[4] == pytest.approx(0.6)
    assert result.score_distribution.expected_score == pytest.approx(3.83)
    assert result.evidence == result.reason == ""
    assert calls[0]["max_tokens"] == 4
    assert calls[0]["logprobs"] is True
    assert calls[0]["top_logprobs"] == 20


def test_v3_rejects_non_neutral_score_without_exact_evidence() -> None:
    raw = json.dumps(
        {
            "answer_id": "answer_0",
            "trait_score": 5,
            "evidence": "invented quote",
            "reason": "Strong target evidence.",
        }
    )
    response = SimpleNamespace(
        id="response",
        usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw))],
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    answer = Answer(answer_id="candidate", text="A neutral answer.")
    row = JudgeInput(prompt_id="scenario", scenario="Scenario", answers=[answer])
    feature = FeatureV2(
        target="target",
        opposite="opposite",
        definition="definition",
        anchors={score: str(score) for score in range(1, 6)},
    )

    with pytest.raises(SchemaFailure):
        scalar_trait_task_v3(
            (row, answer),
            feature_name="feature",
            feature=feature,
            template="{target}\n{opposite}\n{definition}\n{exclusions}\n{scale}",
            client=client,
            model="test",
            provider="test",
            temperature=0,
            max_tokens=10,
            schema_retries=0,
            rubric_version="1",
            prompt_version="prompt",
            prompt_sha256="prompt-hash",
            config_version="1",
            config_sha256="config-hash",
            seed=1,
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
            "evidence": "",
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
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: next(responses))
        )
    )
    parsed, raw, _, response_ids, _ = validated_completion(
        client,
        response_model=ScalarTraitResponseV3,
        validate=lambda _: None,
        schema_retries=1,
        model="test",
        system="test",
        payload={},
        temperature=0,
        max_tokens=10,
    )
    assert parsed.trait_score == 3
    assert len(raw) == 2
    assert response_ids == ["bad", "good"]


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


def test_resume_rejects_incompatible_provenance(tmp_path: Path) -> None:
    output = tmp_path / "scores.jsonl"
    output.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "provenance": {"judge_model": "old-model"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def worker(task: str, dry_run: bool = False) -> str:
        return task

    with pytest.raises(ValueError, match="incompatible provenance"):
        run_tasks(
            ["task-1"],
            worker,
            output=output,
            workers=1,
            resume_provenance={"judge_model": "new-model"},
        )
