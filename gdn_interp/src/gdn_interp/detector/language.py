from collections.abc import Iterable

from lingua import Language, LanguageDetector as LinguaDetector, LanguageDetectorBuilder

LANGUAGES = (
    Language.ENGLISH,
    Language.RUSSIAN,
    Language.FRENCH,
    Language.GERMAN,
    Language.SPANISH,
    Language.ITALIAN,
    Language.PORTUGUESE,
    Language.DUTCH,
    Language.UKRAINIAN,
    Language.POLISH,
)


class LanguageDetector:
    """Detect a target language and its change after steering."""

    def __init__(self, target: str, languages: Iterable[Language] = LANGUAGES) -> None:
        self.target = target.lower()
        languages = tuple(languages)
        targets = {language.iso_code_639_1.name.lower() for language in languages}
        if self.target not in targets:
            raise ValueError(f"target language must be one of {sorted(targets)}")
        self.detector: LinguaDetector = LanguageDetectorBuilder.from_languages(*languages).build()

    def label(self, text: str) -> str:
        if not text.strip():
            return "unknown"
        values = self.detector.compute_language_confidence_values(text)
        return values[0].language.iso_code_639_1.name.lower() if values else "unknown"

    def confidence(self, text: str) -> float:
        if not text.strip():
            return 0.0
        values = self.detector.compute_language_confidence_values(text)
        return values[0].value if values else 0.0

    def detects(self, text: str) -> bool:
        return self.label(text) == self.target

    def score(self, base_text: str, steered_text: str) -> float:
        return float(self.detects(steered_text) - self.detects(base_text))
