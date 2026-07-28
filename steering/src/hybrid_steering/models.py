from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LayerSelection(StrictModel):
    decoder_layer_indices_zero_based: list[int]

    @field_validator("decoder_layer_indices_zero_based")
    @classmethod
    def validate_layer_indices(cls, indices: list[int]) -> list[int]:
        if any(index < 0 for index in indices):
            raise ValueError("decoder layer indices must be non-negative")
        if len(indices) != len(set(indices)):
            raise ValueError("decoder layer indices must be unique")
        return indices


class DirectionManifest(LayerSelection):
    schema_version: Literal["1.0"] = "1.0"
    model_id: str = Field(min_length=1)
    model_revision: str | None = None
    transformers_version: str = Field(min_length=1)
    positive_pole: str = Field(min_length=1)
    negative_pole: str = Field(min_length=1)
    train_example_ids: list[str] = Field(min_length=1)
    decoder_layer_indices_zero_based: list[int] = Field(min_length=1)
    state_shapes: dict[int, list[int]]
    formula: Literal["mean_positive_minus_negative"] = "mean_positive_minus_negative"
    dtype: Literal["float32"] = "float32"

    @model_validator(mode="after")
    def validate_indices_and_shapes(self) -> DirectionManifest:
        indices = self.decoder_layer_indices_zero_based
        if set(indices) != set(self.state_shapes):
            raise ValueError("state_shapes keys must match decoder layer indices")
        if len(self.train_example_ids) != len(set(self.train_example_ids)):
            raise ValueError("train_example_ids must be unique")
        return self


class RunManifest(LayerSelection):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str | None = None
    method: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    alpha: float
    seed: int
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    direction_artifact: str | None = None


class GenerationRecord(LayerSelection):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    method: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    alpha: float
    response: str
    seed: int
    metadata: dict[str, Any] = Field(default_factory=dict)
