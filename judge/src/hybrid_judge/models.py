from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ScoreV2 = Annotated[int, Field(ge=1, le=5)]


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


class FeatureV2(Feature):
    anchors: dict[int, str]

    @model_validator(mode="after")
    def complete_scale(self) -> FeatureV2:
        if set(self.anchors) != {1, 2, 3, 4, 5}:
            raise ValueError("feature anchors must define scores 1 through 5")
        return self


class FeatureConfigV2(StrictModel):
    rubric_version: str
    features: dict[str, FeatureV2]


class GenerationConfig(StrictModel):
    temperature: float = 0
    max_output_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: float = Field(default=240, gt=0)
    max_retries: int = Field(default=4, ge=0)
    schema_retries: int = Field(default=2, ge=0)
    workers: int = Field(default=8, gt=0)


class EvaluationConfig(StrictModel):
    default_feature: str
    scalar_prompt: str
    pairwise_prompt: str


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


class JudgeConfigV2(JudgeConfig):
    require_both_orders: bool = True


class ScalarResponseV2(StrictModel):
    answer_id: str
    trait_score: ScoreV2
    task_fulfillment: ScoreV2
    coherence: ScoreV2
    evidence: list[str] = Field(max_length=2)
    reason: str = Field(max_length=400)


class PairwiseResponseV2(StrictModel):
    trait_winner: Literal["A", "B", "tie"]
    quality_winner: Literal["A", "B", "tie"]
    evidence_A: str = Field(max_length=300)
    evidence_B: str = Field(max_length=300)
    reason: str = Field(max_length=400)


class Usage(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class ProvenanceV2(StrictModel):
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
    schema_attempts: int
    timestamp_utc: str
    usage: Usage
    raw_responses: list[str]


class ScalarResultV2(StrictModel):
    task_id: str
    prompt_id: str
    answer_id: str
    feature: str
    trait_score: ScoreV2
    centered_trait_score: int = Field(ge=-2, le=2)
    task_fulfillment: ScoreV2
    coherence: ScoreV2
    evidence: list[str]
    reason: str
    provenance: ProvenanceV2


class PairwiseResultV2(StrictModel):
    task_id: str
    prompt_id: str
    feature: str
    orientation: Literal["one", "reverse"]
    left_answer_id: str
    right_answer_id: str
    trait_winner: Literal["left", "right", "tie"]
    quality_winner: Literal["left", "right", "tie"]
    evidence_left: str
    evidence_right: str
    reason: str
    provenance: ProvenanceV2


class PairwiseAggregateV2(StrictModel):
    aggregate_id: str
    prompt_id: str
    feature: str
    answer_ids: list[str] = Field(min_length=2, max_length=2)
    orientation_count: int = Field(ge=1, le=2)
    status: Literal["complete", "incomplete"]
    trait_order_consistent: bool | None
    quality_order_consistent: bool | None
    trait_winner_answer_id: str | Literal["tie"] | None
    quality_winner_answer_id: str | Literal["tie"] | None
    task_ids: list[str]
