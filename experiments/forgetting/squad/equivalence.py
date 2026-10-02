"""Factual equivalence of a steered answer and its baseline. Not a concept score."""

from __future__ import annotations

import re

_VERDICT = re.compile(r"\s*<verdict>([01])</verdict>\s*\Z")

PROMPT = """You are a factual-equivalence judge. Compare the model answer with the baseline answer for the given question.

Return `<verdict>1</verdict>` only when both answers give the same factual answer. Answers may use different languages, wording, formatting, or detail. Return `<verdict>0</verdict>` if they conflict, either answer does not answer the question, either answer is factually wrong, or equivalence is uncertain.

Your entire response must be exactly one of these two strings, with no other text:
<verdict>1</verdict>
<verdict>0</verdict>

Question:
{question}

Model answer:
{response}

Baseline answer:
{baseline_response}
"""


def prompt(question: str, response: str, baseline_response: str) -> str:
    return PROMPT.format(question=question, response=response, baseline_response=baseline_response)


def verdict(raw: str | None) -> int | None:
    """1 or 0 when the judge returned the tag. Anything else is missing."""
    if raw is None:
        return None
    match = _VERDICT.fullmatch(raw)
    return int(match.group(1)) if match else None
