from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from .models import FeatureConfigV2, JudgeConfigV2

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: Path, model: type[T]) -> T:
    return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_configs(root: Path) -> tuple[FeatureConfigV2, JudgeConfigV2]:
    return (
        load_yaml(root.parent / "concepts" / "features.yaml", FeatureConfigV2),
        load_yaml(root / "config" / "judge.yaml", JudgeConfigV2),
    )
