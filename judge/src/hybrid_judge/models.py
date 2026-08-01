from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

Score = Annotated[int, Field(ge=1, le=5)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Answer(StrictModel):
    answer_id: str = Field(min_length=1)
    text: str


class JudgeInput(StrictModel):
    prompt_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    answers: list[Answer] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_answer_ids(self) -> JudgeInput:
        ids = [answer.answer_id for answer in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("answer_id must be unique inside a prompt")
        return self


class Feature(StrictModel):
    target: str
    opposite: str
    definition: str
    exclusions: list[str] = Field(default_factory=list)
    anchors: dict[int, str]

    @model_validator(mode="after")
    def complete_scale(self) -> Feature:
        if set(self.anchors) != {1, 2, 3, 4, 5}:
            raise ValueError("feature anchors must define scores 1 through 5")
        return self


class FeatureConfig(StrictModel):
    rubric_version: str
    features: dict[str, Feature]


class GenerationConfig(StrictModel):
    temperature: float = 0
    max_output_tokens: int = Field(default=4096, gt=0)
    top_logprobs: int = Field(default=20, ge=5, le=20)
    timeout_seconds: float = Field(default=240, gt=0)
    max_retries: int = Field(default=4, ge=0)
    schema_retries: int = Field(default=2, ge=0)
    workers: int = Field(default=8, gt=0)


class EvaluationConfig(StrictModel):
    default_feature: str
    prompt: str


class ScaleConfig(StrictModel):
    minimum: int = 0
    maximum: int = 4
    anchors: dict[int, str]


class JudgeConfig(StrictModel):
    config_version: str
    provider: Literal["openrouter"]
    model: str
    base_url: str
    generation: GenerationConfig
    evaluation: EvaluationConfig
    scale: ScaleConfig
    quality_metrics: list[str]


class JudgeResponseV3(RootModel[Score]):
    pass


class Usage(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class TraitScoreDistribution(StrictModel):
    probabilities: dict[int, float]
    expected_score: float = Field(ge=1, le=5)
    chosen_score_probability: float = Field(ge=0, le=1)
    entropy: float = Field(ge=0)
    valid_token_mass: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def complete_scale(self) -> TraitScoreDistribution:
        if set(self.probabilities) != {1, 2, 3, 4, 5}:
            raise ValueError("score probabilities must define scores 1 through 5")
        if abs(sum(self.probabilities.values()) - 1) > 1e-5:
            raise ValueError("score probabilities must sum to one")
        return self


class Provenance(StrictModel):
    judge_model: str
    provider: str
    provider_response_ids: list[str]
    prompt_version: str
    prompt_sha256: str
    rubric_version: str
    config_version: str
    config_sha256: str
    answer_order: list[str]
    seed: int
    temperature: float
    logprobs: bool = False
    top_logprobs: int | None = None
    schema_attempts: int
    timestamp_utc: str
    usage: Usage
    raw_responses: list[str]


class JudgeResultV3(StrictModel):
    task_id: str
    prompt_id: str
    answer_id: str
    feature: str
    trait_score: Score
    centered_trait_score: int = Field(ge=-2, le=2)
    score_distribution: TraitScoreDistribution
    provenance: Provenance
