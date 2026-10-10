"""The full-function HumanEval prompt and completion used by the saved 9B run."""

import re

PROMPT = (
    "Solve this Python task. Return a complete implementation of the requested "
    "function, including its def line. Output Python code only: no Markdown "
    "fences, explanations, or tests.\n\n{stub}"
)


def prompt(stub: str) -> str:
    return PROMPT.format(stub=stub)


def completion(response: str) -> str:
    """Match the saved full-function protocol; remove one enclosing code fence."""
    text = response.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    return "\n\n" + (match.group(1) if match else text) + "\n"
