import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hybrid_judge.cli import arguments
from hybrid_judge.config import load_configs
from hybrid_judge.models import Answer, Feature, JudgeInput, JudgeResponseV3
from hybrid_judge.runner import (
    judge_task_v3,
    read_jsonl,
    render_prompt,
    run_tasks,
    trait_tasks,
)

ROOT = Path(__file__).parents[1]


def test_cli_has_one_standard_mode(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hybrid-judge", "input.jsonl", "output.jsonl"])
    args = arguments()
    assert not hasattr(args, "mode")

    monkeypatch.setattr(
        sys,
        "argv",
        ["hybrid-judge", "input.jsonl", "output.jsonl", "--mode", "pairwise"],
    )
    with pytest.raises(SystemExit):
        arguments()


def test_contracts_and_tasks() -> None:
    features, config = load_configs(ROOT)
    assert set(features.features["french_language"].anchors) == {1, 2, 3, 4, 5}
    assert set(features.features["first_person_voice"].anchors) == {1, 2, 3, 4, 5}
    feature = features.features["optimism"]
    prompt = render_prompt(
        (ROOT / "prompts" / config.evaluation.prompt).read_text(encoding="utf-8"),
        feature,
        feature.anchors,
    )
    rows = read_jsonl(ROOT / "examples" / "input.example.jsonl")

    assert "1:" in prompt and "5:" in prompt
    assert len(trait_tasks(rows)) == 2
    assert config.evaluation.prompt == "judge_v3.txt"


def test_v3_accepts_one_digit_and_records_probabilities() -> None:
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
    feature = Feature(
        target="target",
        opposite="opposite",
        definition="definition",
        anchors={score: str(score) for score in range(1, 6)},
    )

    result = judge_task_v3(
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
        prompt_version="judge_v3.txt",
        prompt_sha256="prompt-hash",
        config_version="1",
        config_sha256="config-hash",
        seed=1,
    )

    assert result.task_id == "judge-v3:1:feature:scenario:candidate"
    assert result.trait_score == 4
    assert result.centered_trait_score == 1
    assert result.score_distribution.probabilities[4] == pytest.approx(0.6)
    assert result.score_distribution.expected_score == pytest.approx(3.83)
    assert calls[0]["max_tokens"] == 4
    assert calls[0]["logprobs"] is True
    assert calls[0]["top_logprobs"] == 20


def test_v3_schema_rejects_non_digit() -> None:
    with pytest.raises(ValidationError):
        JudgeResponseV3.model_validate_json('{"score": 4}')


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
