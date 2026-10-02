import json
from types import SimpleNamespace

import pytest

from hybrid_steering.detect import LANGUAGE_FEATURES
from hybrid_steering.judge import load_guides, parse_judgment, render, score_rows, score_steering
from hybrid_steering.judge.config import repo_root
from hybrid_steering.runtime import import_path


def _label(**overrides) -> str:
    body = {
        "evaluable": True,
        "concept_score": 2,
        "content_quality": 3,
        "flags": ["repetition"],
        "reason": "Clear signs in one paragraph, and the answer still addresses the question.",
    }
    body.update(overrides)
    return json.dumps(body)


def test_every_guide_fills_the_shared_rubric() -> None:
    guides = load_guides()
    assert "optimism" in guides
    assert not (set(guides) & set(LANGUAGE_FEATURES))
    for name, guide in guides.items():
        text = render(name)
        assert "{{concept}}" not in text
        assert "concept_score is an integer 0–4" in text
        assert guide.splitlines()[0] in text


def test_parse_judgment_accepts_the_rubric_and_rejects_drift() -> None:
    parsed = parse_judgment(_label())
    assert parsed.concept_score == 2
    assert parsed.flags == ("repetition",)
    empty = parse_judgment(_label(evaluable=False, concept_score=None, content_quality=0, flags=[]))
    assert empty.concept_score is None
    assert empty.content_quality == 0
    with pytest.raises(ValueError):
        parse_judgment(_label(note="extra"))
    with pytest.raises(ValueError):
        parse_judgment(_label(concept_score=5))
    with pytest.raises(ValueError):
        parse_judgment(_label(evaluable=False, concept_score=0))
    with pytest.raises(ValueError):
        parse_judgment(_label(flags=["repetition", "repetition"]))


def test_score_steering_sends_one_batch_and_keeps_order(monkeypatch) -> None:
    import litellm

    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen: dict = {}

    def fake(*, messages, **kwargs):
        seen["messages"] = messages
        seen["workers"] = kwargs["max_workers"]
        seen["format"] = kwargs["response_format"]
        seen["model"] = kwargs["model"]
        seen["base_url"] = kwargs["base_url"]
        seen["api_key"] = kwargs["api_key"]
        return [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=_label()))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="nope"))]),
        ]

    monkeypatch.setattr(litellm, "batch_completion", fake)
    judgments = score_steering([("Why leaves?", "They fall."), ("q2", "a2")], "optimism")
    assert len(seen["messages"]) == 2
    assert seen["workers"] == 8
    assert seen["format"] == {"type": "json_object"}
    assert seen["model"] == "openai/gpt-4o-mini"
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["api_key"] == "test-key"
    assert "plausible favorable" in seen["messages"][0][0]["content"]
    assert json.loads(seen["messages"][0][1]["content"])["prompt"] == "Why leaves?"
    assert judgments[0] is not None and judgments[0].concept_score == 2
    assert judgments[1] is None


def test_score_steering_drops_a_label_cut_at_the_token_limit(monkeypatch) -> None:
    import litellm

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "local")
    seen: dict = {}

    def fake(*, messages, **kwargs):
        seen.update(kwargs)
        return [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=_label()), finish_reason=reason)
                ]
            )
            for reason in ("stop", "length")
        ]

    monkeypatch.setattr(litellm, "batch_completion", fake)
    judgments = score_steering([("q", "a"), ("q2", "a2")], "optimism", thinking=False)
    assert judgments[0] is not None and judgments[1] is None
    assert seen["max_tokens"] == 4096
    assert seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_score_steering_refuses_a_language_and_a_total_miss(monkeypatch) -> None:
    import litellm

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "local")

    def fake(*, messages, **kwargs):
        return [ValueError("down") for _ in messages]

    monkeypatch.setattr(litellm, "batch_completion", fake)
    with pytest.raises(ValueError, match="unknown feature"):
        score_steering([("q", "a")], "ru")
    with pytest.raises(RuntimeError, match="no scores"):
        score_steering([("q", "a")], "joy")


def test_judge_requires_the_openai_endpoint(monkeypatch) -> None:
    import litellm

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(litellm, "batch_completion", fake)
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        score_steering([("q", "a")], "optimism")


def test_language_rows_do_not_call_the_judge(monkeypatch) -> None:
    import litellm

    def fake(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(litellm, "batch_completion", fake)
    rows = [{"response": "Это длинное русское предложение про дождь, город и море."}]
    score_rows(rows, "ru")
    assert rows[0]["concept_score"] == 1
    assert rows[0]["label"] == "ru"


def test_equivalence_verdict_is_only_the_tag() -> None:
    module = import_path(repo_root() / "experiments/forgetting/squad/equivalence.py")
    assert module.verdict("<verdict>1</verdict>") == 1
    assert module.verdict(" <verdict>0</verdict>\n") == 0
    assert module.verdict("The answers match. <verdict>1</verdict>") is None
    assert module.verdict(None) is None
