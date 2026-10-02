"""Language detection for steering scores.

Natural-language features in ``LANGUAGE_FEATURES`` are scored with Lingua.
Every other concept is scored by ``hybrid_steering.judge.score_steering``.
``EVAL_QUESTIONS`` are held-out English prompts for a steering run.
"""

from __future__ import annotations

from typing import Protocol

from lingua import Language, LanguageDetectorBuilder

LANGUAGES = (
    Language.ENGLISH,
    Language.RUSSIAN,
    Language.FRENCH,
)
# Targets outside LANGUAGES join the candidate set only for their own detector,
# so Russian and French detection keeps exactly the candidates it was scored with.
EXTRA_LANGUAGES = {
    "zh": Language.CHINESE,
    "ar": Language.ARABIC,
    "hi": Language.HINDI,
}

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
# (concrete, technical) are concepts and go through the steering judge.
LANGUAGE_FEATURES = {
    "french_language": "fr",
    "russian_language": "ru",
    "chinese_language": "zh",
    "arabic_language": "ar",
    "hindi_language": "hi",
}


class ConceptDetector(Protocol):
    """Whether a text is in one language."""

    target: str
    source: str

    def label(self, text: str, *, question: str = "") -> str:
        """ISO language code, or ``unknown``."""

    def detects(self, text: str, *, question: str = "") -> bool:
        """True when the text is in the target language."""


def is_language(feature: str) -> bool:
    """True for a Lingua feature id or its ISO code."""
    return feature in LANGUAGE_FEATURES or feature in LANGUAGE_FEATURES.values()


def concept_detector(feature: str) -> ConceptDetector:
    """Score a natural language locally. Other concepts use the steering judge."""
    code = LANGUAGE_FEATURES.get(feature)
    if code is None and feature in LANGUAGE_FEATURES.values():
        code = feature
    if code is None:
        known = ", ".join(sorted({*LANGUAGE_FEATURES, *LANGUAGE_FEATURES.values()}))
        raise ValueError(f"{feature!r} is not a language feature ({known})")
    return _LinguaDetector(code)


class _LinguaDetector:
    """Local language detection. Does not call a model."""

    def __init__(self, target: str, languages: tuple[Language, ...] = LANGUAGES) -> None:
        self.target = target.lower()
        self.source = ""
        if self.target in EXTRA_LANGUAGES:
            languages = (*languages, EXTRA_LANGUAGES[self.target])
        self.languages = languages
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
