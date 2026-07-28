from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .cache import TensorMap, recurrent_tensor


@torch.no_grad()
def add_direction(
    cache: Any,
    direction: TensorMap,
    alpha: float,
    *,
    layers: Iterable[int] | None = None,
) -> None:
    """Add alpha * direction to selected absolute decoder layer indices."""
    for layer_idx in _selected_layers(direction, layers):
        target = recurrent_tensor(cache, layer_idx)
        delta = direction[layer_idx].to(device=target.device, dtype=target.dtype)
        if target.shape != delta.shape:
            raise ValueError(f"direction shape mismatch at layer {layer_idx}")
        target.add_(delta, alpha=float(alpha))


@torch.no_grad()
def replace_recurrent(
    cache: Any,
    donor: TensorMap,
    *,
    layers: Iterable[int] | None = None,
) -> None:
    """Replace selected recurrent states without touching KV or convolution state."""
    for layer_idx in _selected_layers(donor, layers):
        target = recurrent_tensor(cache, layer_idx)
        source = donor[layer_idx].to(device=target.device, dtype=target.dtype)
        if target.shape != source.shape:
            raise ValueError(f"donor shape mismatch at layer {layer_idx}")
        target.copy_(source)


def _selected_layers(
    values: TensorMap, layers: Iterable[int] | None
) -> tuple[int, ...]:
    selected = tuple(values) if layers is None else tuple(layers)
    if len(selected) != len(set(selected)):
        raise ValueError("decoder layer indices must be unique")
    for layer_idx in selected:
        if layer_idx < 0:
            raise ValueError("decoder layer indices are zero-based and non-negative")
        if layer_idx not in values:
            raise KeyError(f"no tensor for decoder layer {layer_idx}")
    return selected
