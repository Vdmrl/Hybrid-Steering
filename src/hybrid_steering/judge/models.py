"""Plain records for judge inputs, config, and scores."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Answer:
    answer_id: str
    text: str


@dataclass
class JudgeInput:
    prompt_id: str
    scenario: str
    answers: list[Answer]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Feature:
    target: str
    opposite: str
    definition: str
    anchors: dict[int, str]
    exclusions: list[str] = field(default_factory=list)


@dataclass
class FeatureConfig:
    features: dict[str, Feature]


@dataclass
class GenerationConfig:
    temperature: float = 0
    max_output_tokens: int = 4096
    top_logprobs: int = 20
    timeout_seconds: float = 240
    retries: int = 2
    workers: int = 8
    request_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    default_feature: str
    prompt: str


@dataclass
class JudgeConfig:
    model: str
    base_url: str
    generation: GenerationConfig
    evaluation: EvaluationConfig


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class ScoreDistribution:
    probabilities: dict[int, float]
    expected_score: float
    chosen_score_probability: float
    entropy: float
    valid_token_mass: float


@dataclass
class JudgeResult:
    task_id: str
    prompt_id: str
    answer_id: str
    feature: str
    trait_score: int
    centered_trait_score: int
    score_distribution: ScoreDistribution
    model: str
    prompt: str
    raw: str
    usage: Usage

    def json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
