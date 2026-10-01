from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class JudgeSettings:
    """Model id and generation limits for the OpenAI-compatible endpoint.

    ``max_tokens`` covers the reasoning trace when ``thinking`` is on: a trace
    that hits the limit leaves no label.
    """

    model: str
    max_tokens: int = 4096
    thinking: bool = True


def repo_root() -> Path:
    """Repository root: the directory that holds concepts/ and config/."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "concepts" / "features.yaml").is_file() and (
            parent / "config" / "judge.yaml"
        ).is_file():
            return parent
    raise FileNotFoundError("concepts/features.yaml and config/judge.yaml were not found")


def load_settings(root: Path | None = None) -> JudgeSettings:
    """Read ``config/judge.yaml``. Only ``model`` is required.

    The endpoint and key are ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``.
    """
    root = root or repo_root()
    raw = yaml.safe_load((root / "config" / "judge.yaml").read_text(encoding="utf-8"))
    defaults = JudgeSettings(model="")
    return JudgeSettings(
        model=str(raw["model"]),
        max_tokens=int(raw.get("max_tokens", defaults.max_tokens)),
        thinking=bool(raw.get("thinking", defaults.thinking)),
    )
