import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_steering.judge.cli import arguments
from hybrid_steering.judge.config import load_configs
from hybrid_steering.judge.models import Answer, Feature, JudgeInput
from hybrid_steering.judge.runner import (
    complete_text,
    judge_task_v3,
    parse_score,
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
    assert set(features.features["bulleted_layout"].anchors) == {1, 2, 3, 4, 5}
    feature = features.features["optimism"]
    prompt = render_prompt(
        (ROOT / "prompts" / config.evaluation.prompt).read_text(encoding="utf-8"),
        feature,
        feature.anchors,
    )
    rows = [
        JudgeInput(
            prompt_id="scenario-001",
            scenario="A team must decide whether to continue a risky project after mixed evidence.",
            answers=[
                Answer("baseline", "Review the evidence and decide tomorrow."),
                Answer("steered", "Everything will probably fail."),
            ],
        )
    ]

    assert "1:" in prompt and "5:" in prompt
    assert len(trait_tasks(rows)) == 2
    assert config.evaluation.prompt == "judge_v3_compositional.txt"


def test_compositional_prompt_renders_for_every_feature() -> None:
    features, _ = load_configs(ROOT)
    template = (ROOT / "prompts" / "judge_v3_compositional.txt").read_text(encoding="utf-8")

    for feature in features.features.values():
        prompt = render_prompt(template, feature, feature.anchors)
        assert "Feature-specific anchored scale:" in prompt
        assert "anchors; they are authoritative" in prompt
        assert "Return exactly one ASCII digit from 1 to 5." in prompt
        assert "{target}" not in prompt
        assert all(f"{score}:" in prompt for score in range(1, 6))


def _digit_response(content: str = "4"):
    probabilities = {1: 0.02, 2: 0.03, 3: 0.20, 4: 0.60, 5: 0.15}
    top_logprobs = [
        SimpleNamespace(token=str(score), logprob=math.log(probability))
        for score, probability in probabilities.items()
    ]
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                logprobs=SimpleNamespace(
                    content=[SimpleNamespace(token=content, top_logprobs=top_logprobs)]
                ),
            )
        ],
    )


def test_v3_accepts_one_digit_and_records_probabilities(monkeypatch) -> None:
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _digit_response()

    monkeypatch.setattr("litellm.completion", completion)
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
        model="test",
        temperature=0,
        max_tokens=256,
        top_logprobs=20,
        retries=0,
        prompt_name="judge_v3.txt",
    )

    assert result.task_id == "feature:scenario:candidate"
    assert result.trait_score == 4
    assert result.centered_trait_score == 1
    assert result.score_distribution.probabilities[4] == pytest.approx(0.6)
    assert result.score_distribution.expected_score == pytest.approx(3.83)
    assert calls[0]["max_tokens"] == 4
    assert calls[0]["logprobs"] is True
    assert calls[0]["top_logprobs"] == 20


def test_complete_reads_key_and_proxy_without_printing(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    monkeypatch.setenv("OPENROUTER_PROXY", "http://proxy.example")
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    seen = {}

    def completion(**kwargs):
        seen.update(kwargs)
        return _digit_response("1")

    monkeypatch.setattr("litellm.completion", completion)
    assert complete_text("hello", model="test-model") == "1"
    assert seen["api_key"] == "secret-key"
    assert os.environ["HTTPS_PROXY"] == "http://proxy.example"
    captured = capsys.readouterr()
    assert "secret-key" not in captured.out
    assert "proxy.example" not in captured.out


def test_parse_score_rejects_json() -> None:
    with pytest.raises(ValueError):
        parse_score('{"score": 4}')


def test_queue_persists_failures(tmp_path: Path) -> None:
    output = tmp_path / "scores.jsonl"

    def worker(task: str, dry_run: bool = False) -> str:
        if dry_run:
            return task
        raise ValueError("invalid response")

    assert run_tasks(["task-1"], worker, output=output, workers=1) == (0, 1)
    failure = json.loads((tmp_path / "scores.failures.jsonl").read_text(encoding="utf-8"))
    assert failure["task_id"] == "task-1"


def test_resume_skips_finished_task_ids(tmp_path: Path) -> None:
    output = tmp_path / "scores.jsonl"
    output.write_text(json.dumps({"task_id": "task-1"}) + "\n", encoding="utf-8")
    calls = []

    def worker(task: str, dry_run: bool = False) -> str:
        if dry_run:
            return task
        calls.append(task)
        return task

    assert run_tasks(["task-1"], worker, output=output, workers=1) == (0, 0)
    assert calls == []
