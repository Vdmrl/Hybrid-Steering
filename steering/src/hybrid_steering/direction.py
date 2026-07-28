from __future__ import annotations

from collections.abc import Iterable

import torch

from .cache import TensorMap


def subtract_states(positive: TensorMap, negative: TensorMap) -> TensorMap:
    """Compute positive minus negative recurrent state on CPU/FP32."""
    _same_layers(positive, negative)
    result: TensorMap = {}
    for layer_idx in positive:
        if positive[layer_idx].shape != negative[layer_idx].shape:
            raise ValueError(f"state shape mismatch at layer {layer_idx}")
        result[layer_idx] = (
            positive[layer_idx].detach().float().cpu()
            - negative[layer_idx].detach().float().cpu()
        )
    return result


def mean_direction(differences: Iterable[TensorMap]) -> TensorMap:
    """Average paired state differences on CPU/FP32."""
    sums: TensorMap = {}
    count = 0
    expected_layers: set[int] | None = None
    for difference in differences:
        layers = set(difference)
        if expected_layers is None:
            expected_layers = layers
        elif layers != expected_layers:
            raise ValueError("direction samples contain different layers")
        for layer_idx, tensor in difference.items():
            tensor = tensor.detach().float().cpu()
            if layer_idx in sums and sums[layer_idx].shape != tensor.shape:
                raise ValueError(f"direction shape mismatch at layer {layer_idx}")
            sums[layer_idx] = sums.get(layer_idx, torch.zeros_like(tensor)) + tensor
        count += 1
    if count == 0:
        raise ValueError("cannot average zero state differences")
    return {layer_idx: tensor / count for layer_idx, tensor in sums.items()}


def _same_layers(left: TensorMap, right: TensorMap) -> None:
    if set(left) != set(right):
        raise ValueError("recurrent states contain different decoder layers")
