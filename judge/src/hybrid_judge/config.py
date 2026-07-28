from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from .models import FeatureConfig, FeatureConfigV2, JudgeConfig, JudgeConfigV2

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: Path, model: type[T]) -> T:
    return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_configs(
    root: Path, version: str = "v2"
) -> tuple[FeatureConfig | FeatureConfigV2, JudgeConfig | JudgeConfigV2]:
    if version == "v1":
        return (
            load_yaml(root.parent / "concepts" / "features_v1.yaml", FeatureConfig),
            load_yaml(root / "config" / "judge_v1.yaml", JudgeConfig),
        )
    if version == "v2":
        return (
            load_yaml(root.parent / "concepts" / "features_v2.yaml", FeatureConfigV2),
            load_yaml(root / "config" / "judge_v2.yaml", JudgeConfigV2),
        )
    raise ValueError(f"unknown judge version: {version}")
