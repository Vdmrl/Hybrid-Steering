from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (
    EvaluationConfig,
    Feature,
    FeatureConfig,
    GenerationConfig,
    JudgeConfig,
)


def repo_root() -> Path:
    """Repository root: the directory that holds concepts/ and config/."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "concepts" / "features.yaml").is_file() and (
            parent / "config" / "judge.yaml"
        ).is_file():
            return parent
    raise FileNotFoundError("concepts/features.yaml and config/judge.yaml were not found")


def load_judge_config(path: Path) -> JudgeConfig:
    return _judge_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_configs(root: Path) -> tuple[FeatureConfig, JudgeConfig]:
    features = yaml.safe_load((root / "concepts" / "features.yaml").read_text(encoding="utf-8"))
    return _features(features), load_judge_config(root / "config" / "judge.yaml")


def _features(raw: dict[str, Any]) -> FeatureConfig:
    features = {}
    for name, item in raw["features"].items():
        anchors = {int(score): text for score, text in item["anchors"].items()}
        if set(anchors) != {1, 2, 3, 4, 5}:
            raise ValueError(f"{name} anchors must be scores 1 through 5")
        features[name] = Feature(
            target=item["target"],
            opposite=item["opposite"],
            definition=item["definition"],
            anchors=anchors,
            exclusions=list(item.get("exclusions") or []),
        )
    return FeatureConfig(features=features)


def _judge_config(raw: dict[str, Any]) -> JudgeConfig:
    generation = raw.get("generation") or {}
    evaluation = raw["evaluation"]
    return JudgeConfig(
        model=raw["model"],
        base_url=raw["base_url"],
        generation=GenerationConfig(
            temperature=generation.get("temperature", 0),
            max_output_tokens=generation.get("max_output_tokens", 4096),
            top_logprobs=generation.get("top_logprobs", 20),
            timeout_seconds=generation.get("timeout_seconds", 240),
            retries=generation.get("retries", 2),
            workers=generation.get("workers", 8),
            request_extras=dict(generation.get("request_extras") or {}),
        ),
        evaluation=EvaluationConfig(
            default_feature=evaluation["default_feature"],
            prompt=evaluation["prompt"],
        ),
    )
