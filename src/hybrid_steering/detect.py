"""One concept detector.

Natural-language features in ``LANGUAGE_FEATURES`` are scored with Lingua.
Every other feature is scored from its ``concepts/features.yaml`` definition,
through a 0/1 prompt. The anchored 1–5 judge is a separate article path.
Answer equivalence is not a concept score: it asks whether two answers state
the same fact. ``QUESTIONS`` are the held-out English prompts for that score.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from lingua import Language, LanguageDetectorBuilder

LANGUAGES = (
    Language.ENGLISH,
    Language.RUSSIAN,
    Language.FRENCH,
)

EVAL_QUESTIONS = (
    "Why do leaves fall in autumn?",
    "How can I remove a coffee stain from a shirt?",
    "What is the purpose of a computer's memory?",
    "Why does bread rise?",
    "How do bees communicate?",
    "What is the difference between a lake and a river?",
    "How can I keep cut flowers fresh?",
    "Why do we see lightning before hearing thunder?",
    "What makes a good password?",
    "How does a refrigerator keep food cold?",
    "Why do onions make people cry?",
    "How can I organize a small desk?",
    "What is the difference between a planet and a star?",
    "Why does soap make cleaning easier?",
    "How does a parachute slow someone down?",
    "What is an algorithm?",
    "Why is the sea salty?",
    "How can I learn to play the guitar?",
    "What causes a rainbow?",
    "Why does a compass point north?",
)

# Feature ids whose target is a natural language. Other ``*_language`` features
# (concrete, technical) are concepts and go through the judge.
LANGUAGE_FEATURES = {
    "french_language": "fr",
    "russian_language": "ru",
}

VERDICT_PATTERN = re.compile(r"\s*<verdict>([01])</verdict>\s*\Z")

EQUIVALENCE_PROMPT = """You are a factual-equivalence judge. Compare the model answer with the baseline answer for the given question.

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


class ConceptDetector(Protocol):
    """Whether a text expresses one feature."""

    target: str
    source: str

    def label(self, text: str, *, question: str = "") -> str:
        """Language code, or ``0``/``1`` for a judged concept."""

    def detects(self, text: str, *, question: str = "") -> bool:
        """True when the text expresses the target."""


def concept_detector(
    feature: str,
    *,
    verdict: str | None = None,
    model: str | None = None,
) -> ConceptDetector:
    """Score ``feature`` with Lingua or with the concept judge.

    ``feature`` is a ``concepts/features.yaml`` id, or an ISO code from
    ``LANGUAGE_FEATURES`` (``ru``, ``fr``). ``verdict`` skips the model and
    returns that constant 0/1 string. Used by squeezed local runs.
    """
    code = LANGUAGE_FEATURES.get(feature)
    if code is None and feature in LANGUAGE_FEATURES.values():
        code = feature
    if code is not None:
        return _LinguaDetector(code)
    return _PromptDetector(feature, verdict=verdict, model=model)


class _LinguaDetector:
    """Local language detection. Does not call a model."""

    def __init__(self, target: str, languages: tuple[Language, ...] = LANGUAGES) -> None:
        self.target = target.lower()
        self.source = ""
        names = {language.iso_code_639_1.name.lower() for language in languages}
        if self.target not in names:
            raise ValueError(f"target language must be one of {sorted(names)}")
        self._detector = LanguageDetectorBuilder.from_languages(*languages).build()

    def label(self, text: str, *, question: str = "") -> str:
        del question
        if not text.strip():
            return "unknown"
        values = self._detector.compute_language_confidence_values(text)
        return values[0].language.iso_code_639_1.name.lower() if values else "unknown"

    def detects(self, text: str, *, question: str = "") -> bool:
        del question
        return self.label(text) == self.target


class _PromptDetector:
    """0/1 score from the feature definition in ``concepts/features.yaml``."""

    def __init__(self, feature: str, *, verdict: str | None, model: str | None) -> None:
        from hybrid_steering.judge.config import load_configs, repo_root

        features, config = load_configs(repo_root())
        try:
            spec = features.features[feature]
        except KeyError:
            known = ", ".join(sorted(features.features))
            raise ValueError(f"unknown feature {feature!r}; choose from {known}") from None
        self.target = spec.target
        self.source = spec.opposite
        self._prompt = binary_prompt(spec.target, spec.opposite, spec.definition)
        self._verdict = verdict
        self._model = model or config.model
        self._base_url = config.base_url
        self._extra = config.generation.request_extras

    def label(self, text: str, *, question: str = "") -> str:
        return str(self._score(text, question))

    def detects(self, text: str, *, question: str = "") -> bool:
        return self._score(text, question) == 1

    def _score(self, text: str, question: str) -> int:
        if self._verdict is not None:
            return parse_binary(self._verdict)
        from hybrid_steering.judge.runner import complete_text

        raw = complete_text(
            self._prompt.format(question=question, response=text),
            model=self._model,
            base_url=self._base_url,
            max_tokens=8,
            extra=self._extra,
        )
        return parse_binary(raw)


def binary_prompt(target: str, opposite: str, definition: str) -> str:
    """0/1 prompt for a non-language concept. The definition comes from the rubric."""
    return (
        f"Decide whether the answer shows {target} rather than {opposite}.\n\n"
        f"{definition.strip()}\n\n"
        "Score 1 when the answer matches the target. "
        "Score 0 when it matches the opposite, or when the distinction is unclear.\n\n"
        "Return exactly 0 or 1 and nothing else.\n\n"
        "Question:\n{question}\n\n"
        "Answer:\n{response}"
    )


def parse_binary(raw: str) -> int:
    text = raw.strip()
    if text in {"0", "1"}:
        return int(text)
    match = VERDICT_PATTERN.fullmatch(raw)
    return int(match.group(1)) if match else 0


class AnswerEquivalence:
    """Ask whether a steered answer states the same fact as the baseline."""

    def prompt(self, question: str, response: str, baseline_response: str) -> str:
        return EQUIVALENCE_PROMPT.format(
            question=question,
            response=response,
            baseline_response=baseline_response,
        )

    def score(
        self,
        question: str,
        response: str,
        baseline_response: str,
        judge: Callable[[str], str],
    ) -> tuple[int, str]:
        raw = judge(self.prompt(question, response, baseline_response))
        match = VERDICT_PATTERN.fullmatch(raw)
        return (int(match.group(1)) if match else 0), raw
