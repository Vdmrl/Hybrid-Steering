"""Send blinded Judge requests and derive concept probabilities from token logprobs."""

import json
import math
import time
from pathlib import Path
from typing import Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .io import now, write
from hybrid_steering.paths import ROOT

FLAGS = {
    "empty", "repetition", "incoherent", "off_topic", "unsupported_claim",
    "apparent_truncation", "quotation_only", "prompt_injection_attempt",
}
DETAILS_INSTRUCTION = (
    'Return exactly one JSON object with keys evaluable (boolean), content_quality '
    '(integer 0-4), flags (array), and reason (at most 35 words explaining concept '
    'expression and content quality). Do not include concept_score or probabilities. '
    'For an empty or wholly unintelligible answer set evaluable=false.'
)
SCORE_INSTRUCTION = (
    'Rate the target concept. Return exactly one ASCII digit from 0 to 4 as the '
    'first and only token. No leading whitespace, punctuation, explanation or JSON.'
)
STRUCTURED_SCORE_INSTRUCTION = (
    'Rate the target concept. Return only a JSON object with one integer field '
    '"score" whose value is 0, 1, 2, 3, or 4.'
)
T = TypeVar("T")


class IncompleteLogprobsError(ValueError):
    """The provider did not expose enough score-token probabilities."""


def validate_details(details: dict) -> None:
    """Check model-produced metadata before combining it with logprob scores."""
    assert set(details) == {"evaluable", "content_quality", "flags", "reason"}
    assert type(details["evaluable"]) is bool
    assert type(details["content_quality"]) is int and 0 <= details["content_quality"] <= 4
    assert isinstance(details["flags"], list)
    assert all(isinstance(flag, str) for flag in details["flags"])
    assert len(set(details["flags"])) == len(details["flags"])
    assert set(details["flags"]) <= FLAGS
    assert isinstance(details["reason"], str)
    assert 0 < len(details["reason"].split()) <= 35


def validate(label: dict) -> None:
    """Require the common 0-4 schema; missing judgments are never zero scores."""
    probabilities = tuple(f"p{score}" for score in range(5))
    assert set(label) == {"evaluable", "concept_score", "content_quality", "flags", "reason", *probabilities}
    validate_details({key: label[key] for key in ("evaluable", "content_quality", "flags", "reason")})
    if label["evaluable"]:
        assert type(label["concept_score"]) is int and 0 <= label["concept_score"] <= 4
        values = [label[name] for name in probabilities]
        assert all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values)
        assert abs(sum(values) - 1) <= 1e-6
        assert values[label["concept_score"]] == max(values)
    else:
        assert label["concept_score"] is None
        assert all(label[name] is None for name in probabilities)


def payload(api: dict, system: str, row: dict, kind: str = "details") -> dict:
    """Expose only prompt/response; score request demands actual token logprobs."""
    assert kind in {"details", "score"}
    common = {"model": api["model"], "temperature": api["temperature"],
              "provider": {"require_parameters": True}}
    if kind == "details":
        options = {"max_tokens": api["max_tokens"], "response_format": api["response_format"]}
        instruction = DETAILS_INSTRUCTION
    else:
        options = {"max_tokens": api["score_max_tokens"], "logprobs": True,
                   "top_logprobs": api["score_top_logprobs"]}
        structured = api.get("score_response_format")
        if structured:
            options["response_format"] = structured
        instruction = STRUCTURED_SCORE_INSTRUCTION if structured else SCORE_INSTRUCTION
    return {
        **common, **options,
        "messages": [
            {"role": "system", "content": system + "\n\n" + instruction},
            {"role": "user", "content": json.dumps({
                "prompt": row["prompt"], "response": row["response"],
            }, ensure_ascii=False)},
        ],
    }


def _parse_details(choice: dict) -> dict:
    assert choice["finish_reason"] == "stop"
    details = json.loads(choice["message"]["content"])
    validate_details(details)
    return details


def probabilities_from_score_token(token: dict, digit: str) -> dict[str, float]:
    """Conditional 0-4 probabilities from the token that emitted the score."""
    if token.get("token") != digit:
        raise ValueError("Generated score is not a complete token")
    alternatives = token.get("top_logprobs")
    if not isinstance(alternatives, list):
        raise IncompleteLogprobsError("Provider omitted score-token alternatives")
    scores = {}
    for candidate in alternatives:
        value = candidate.get("logprob")
        candidate_token = candidate.get("token")
        if candidate_token not in {"0", "1", "2", "3", "4"}:
            continue
        if candidate_token in scores or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Duplicate or invalid score-token logprob")
        scores[candidate_token] = value
    if set(scores) != {"0", "1", "2", "3", "4"}:
        raise IncompleteLogprobsError("Provider omitted at least one 0-4 token logprob")
    offset = max(scores.values())
    weights = [math.exp(scores[str(index)] - offset) for index in range(5)]
    total = sum(weights)
    probabilities = {f"p{index}": weight / total for index, weight in enumerate(weights)}
    if max(range(5), key=lambda index: probabilities[f"p{index}"]) != int(digit):
        raise ValueError("Generated score disagrees with returned token logprobs")
    return probabilities


def probabilities_from_choice(choice: dict) -> tuple[int, dict[str, float]]:
    """Softmax the five one-token labels, conditional on the 0-4 alternatives."""
    digit = choice["message"]["content"]
    if digit not in {"0", "1", "2", "3", "4"}:
        raise ValueError("Score response is not one ASCII digit")
    try:
        first = choice["logprobs"]["content"][0]
        alternatives = first["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        raise IncompleteLogprobsError("Provider omitted score-token logprobs") from None
    if first["token"] != digit:
        raise ValueError("Generated score is not the first complete token")
    return int(digit), probabilities_from_score_token(first, digit)

def _request(
    api: dict, system: str, row: dict, secret: str, output: Path,
    kind: str, parse: Callable[[dict], T],
) -> tuple[dict, Path, T]:
    body = json.dumps(payload(api, system, row, kind)).encode()
    for attempt in range(3):
        try:
            request = Request(
                api["base_url"] + "/chat/completions",
                data=body,
                headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
            )
            with urlopen(request, timeout=90) as response:
                raw = json.load(response)
            # Never persist request headers or the API key.
            saved = {key: raw.get(key) for key in ("id", "model", "provider", "usage", "choices")}
            raw_path = output / "raw" / f'{row["anonymous_id"]}-{kind}-{time.time_ns()}-{attempt}.json'
            write(raw_path, saved)
            assert raw.get("model", "").startswith(api["model"]), "Unexpected judge model"
            return raw, raw_path, parse(raw["choices"][0])
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {error.code}; response body omitted") from None
        except IncompleteLogprobsError as error:
            raise RuntimeError(str(error)) from None
        except (URLError, TimeoutError, ValueError, AssertionError, KeyError, IndexError, TypeError):
            pass  # Server errors can contain credentials; keep logs minimal.
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge {kind} request failed after 3 attempts; label remains missing")


def _raw_name(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def request_label(api: dict, system: str, row: dict, secret: str, output: Path) -> dict:
    """Get quality metadata, then a one-token concept score with logprobs."""
    assert api["base_url"] == "https://openrouter.ai/api/v1"
    details_raw, details_path, details = _request(
        api, system, row, secret, output, "details", _parse_details)
    score_raw = None
    score_path = None
    concept_score = None
    probabilities = {f"p{index}": None for index in range(5)}
    if details["evaluable"]:
        score_raw, score_path, scored = _request(
            api, system, row, secret, output, "score", probabilities_from_choice)
        concept_score, probabilities = scored
    label = {**details, "concept_score": concept_score, **probabilities}
    validate(label)
    return {
        "anonymous_id": row["anonymous_id"], **label,
        "probability_source": "token_logprobs_conditional_0_4" if score_raw else None,
        "requested_model": api["model"],
        "returned_model": details_raw["model"],
        "score_returned_model": score_raw["model"] if score_raw else None,
        "provider": details_raw.get("provider"),
        "score_provider": score_raw.get("provider") if score_raw else None,
        "request_id": details_raw.get("id"),
        "score_request_id": score_raw.get("id") if score_raw else None,
        "usage": details_raw.get("usage"),
        "score_usage": score_raw.get("usage") if score_raw else None,
        "created_at": now(),
        "raw_file": _raw_name(details_path),
        "score_raw_file": _raw_name(score_path) if score_path else None,
    }
