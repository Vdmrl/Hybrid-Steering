"""One-call concept-strength Judge; no content-quality assessment."""

import json

import math
import re
from pathlib import Path

from .io import now
from .openrouter import IncompleteLogprobsError, _raw_name, _request, probabilities_from_choice, probabilities_from_score_token

PROBABILITY_SOURCE = "token_logprobs_conditional_0_4"
MISSING_SOURCE = "not_available_in_provider_top_logprobs"
STRUCTURED_SOURCE = "structured_value_token_logprobs_conditional_0_4"
STRUCTURED_MISSING_SOURCE = "structured_value_token_logprobs_incomplete"


def parse_score(choice: dict, *, structured: bool = False) -> tuple[int, dict[str, float | None], str]:
    """Keep a valid score; never infer missing probabilities."""
    if choice.get("finish_reason") != "stop":
        raise ValueError("Judge score did not finish normally")
    if structured:
        answer = json.loads(choice.get("message", {}).get("content", ""))
        if (not isinstance(answer, dict) or set(answer) != {"score"}
                or type(answer["score"]) is not int or not 0 <= answer["score"] <= 4):
            raise ValueError("Structured Judge score must be an integer 0-4")
        content = choice["message"]["content"]
        match = re.search(r'"score"\s*:\s*([0-4])(?=\s*[,}])', content)
        parts = choice.get("logprobs", {}).get("content")
        if not match or not isinstance(parts, list) or "".join(part.get("token", "") for part in parts) != content:
            return answer["score"], {f"p{i}": None for i in range(5)}, STRUCTURED_MISSING_SOURCE
        offset = 0
        for part in parts:
            if offset == match.start(1):
                try:
                    probabilities = probabilities_from_score_token(part, match.group(1))
                except IncompleteLogprobsError:
                    break
                return answer["score"], probabilities, STRUCTURED_SOURCE
            offset += len(part["token"])
        return answer["score"], {f"p{i}": None for i in range(5)}, STRUCTURED_MISSING_SOURCE
    digit = choice.get("message", {}).get("content")
    if digit not in {"0", "1", "2", "3", "4"}:
        raise ValueError("Judge response must be exactly one ASCII digit 0-4")
    try:
        score, probabilities = probabilities_from_choice(choice)
    except IncompleteLogprobsError:
        return int(digit), {f"p{i}": None for i in range(5)}, MISSING_SOURCE
    return score, probabilities, PROBABILITY_SOURCE


def validate(label: dict) -> None:
    """A score is mandatory; probabilities are either complete or all null."""
    required = {"concept_score", "probability_source", *(f"p{i}" for i in range(5))}
    if set(label) != required or type(label["concept_score"]) is not int:
        raise ValueError("Invalid concept-only label schema")
    if not 0 <= label["concept_score"] <= 4:
        raise ValueError("Concept score must be 0-4")
    values = [label[f"p{i}"] for i in range(5)]
    if label["probability_source"] in {MISSING_SOURCE, STRUCTURED_MISSING_SOURCE}:
        if any(value is not None for value in values):
            raise ValueError("Incomplete logprobs must not become probabilities")
    elif label["probability_source"] in {PROBABILITY_SOURCE, STRUCTURED_SOURCE}:
        if (not all(type(value) in (int, float) and math.isfinite(value)
                    and 0 <= value <= 1 for value in values)
                or abs(sum(values) - 1) > 1e-6
                or values[label["concept_score"]] != max(values)):
            raise ValueError("Invalid complete token-logprob distribution")
    else:
        raise ValueError("Unknown probability provenance")


def request_label(api: dict, system: str, row: dict, secret: str, output: Path) -> dict:
    """Request only a concept digit; save raw metadata without the secret."""
    if api["base_url"] != "https://openrouter.ai/api/v1":
        raise ValueError("Unexpected Judge API endpoint")
    raw, raw_path, (score, probabilities, source) = _request(
        api, system, row, secret, output, "score",
        lambda choice: parse_score(choice, structured=bool(api.get("score_response_format"))))
    label = {"concept_score": score, **probabilities, "probability_source": source}
    validate(label)
    return {
        "anonymous_id": row["anonymous_id"], **label,
        "requested_model": api["model"], "returned_model": raw["model"],
        "provider": raw.get("provider"), "request_id": raw.get("id"),
        "usage": raw.get("usage"), "created_at": now(),
        "raw_file": _raw_name(raw_path),
    }
