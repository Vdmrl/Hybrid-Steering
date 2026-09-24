from typing import Protocol


class ConceptDetector(Protocol):
    """Measure a concept's change between base and steered text."""

    def score(self, base_text: str, steered_text: str) -> float: ...
