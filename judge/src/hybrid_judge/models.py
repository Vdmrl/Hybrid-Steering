from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Score = Annotated[int, Field(ge=0, le=4)]


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


class FeatureConfig(StrictModel):
    rubric_version: str
    features: dict[str, Feature]


class GenerationConfig(StrictModel):
    temperature: float = 0
    max_output_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: float = Field(default=240, gt=0)
    max_retries: int = Field(default=4, ge=0)
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


class ScalarScore(StrictModel):
    answer_id: str
    target_score: Score
    opposite_score: Score
    task_correctness: Score
    coherence: Score
    content_preservation: Score
    reason: str


class ScalarResponse(StrictModel):
    scores: list[ScalarScore]


class PairwiseResponse(StrictModel):
    feature_winner: Literal["A", "B", "tie"]
    quality_winner: Literal["A", "B", "tie"]
    reason: str


class Usage(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class Provenance(StrictModel):
    judge_model: str
    prompt_version: str
    rubric_version: str
    permutation: list[str]
    usage: Usage


class ScalarResult(StrictModel):
    task_id: str
    prompt_id: str
    feature: str
    scores: list[ScalarScore]
    provenance: Provenance


class PairwiseResult(StrictModel):
    task_id: str
    prompt_id: str
    feature: str
    left_answer_id: str
    right_answer_id: str
    feature_winner: Literal["left", "right", "tie"]
    quality_winner: Literal["left", "right", "tie"]
    reason: str
    provenance: Provenance
