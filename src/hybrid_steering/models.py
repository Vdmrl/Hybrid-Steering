"""Labels stored next to a direction tensor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DirectionManifest:
    """What a saved target-minus-source direction was computed from."""

    model_id: str
    target: str
    source: str
    example_ids: list[str]
    decoder_layer_indices: list[int]
    state_shapes: dict[int, list[int]]
    rank: int | None = None
