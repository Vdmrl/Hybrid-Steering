"""Blind 1–5 judge. It scores an answer and a feature. It does not see the steering method."""

from .cli import main
from .config import load_configs, load_judge_config
from .models import Feature, FeatureConfig, JudgeConfig, JudgeResult

__all__ = [
    "Feature",
    "FeatureConfig",
    "JudgeConfig",
    "JudgeResult",
    "load_configs",
    "load_judge_config",
    "main",
]
