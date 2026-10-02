from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class JudgeSettings:
    """Model id sent to the OpenAI-compatible endpoint."""

    model: str


def repo_root() -> Path:
    """Repository root: the directory that holds concepts/ and config/."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "concepts" / "features.yaml").is_file() and (
            parent / "config" / "judge.yaml"
        ).is_file():
            return parent
    raise FileNotFoundError("concepts/features.yaml and config/judge.yaml were not found")


def load_settings(root: Path | None = None) -> JudgeSettings:
    """Read the model id from ``config/judge.yaml``.

    The endpoint and key are ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``.
    """
    root = root or repo_root()
    raw = yaml.safe_load((root / "config" / "judge.yaml").read_text(encoding="utf-8"))
    return JudgeSettings(model=str(raw["model"]))
