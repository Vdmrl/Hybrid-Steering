"""Tensor operations on GDN recurrent states.

A direction is ``target - source``: the displacement from the source concept
toward the target concept. Stored directions are float32 maps keyed by absolute
decoder-layer index.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol

import torch
from jaxtyping import Float, Float32, Int
from torch import Tensor

TensorMap = dict[int, Tensor]
Scale = float | Float[Tensor, ""] | Float[Tensor, "batch"]
BatchIndex = Int[Tensor, "batch"]
HeadState = Float[Tensor, "heads key value"]
BatchState = Float[Tensor, "batch heads key value"]


def difference(
    target: dict[int, HeadState], source: dict[int, HeadState]
) -> dict[int, Float32[Tensor, "heads key value"]]:
    """Return target minus source on CPU in float32."""
    _same_layers(target, source)
    result: TensorMap = {}
    for layer in target:
        if target[layer].shape != source[layer].shape:
            raise ValueError(f"state shape mismatch at layer {layer}")
        result[layer] = target[layer].detach().float().cpu() - source[layer].detach().float().cpu()
    return result


def mean(samples: Iterable[dict[int, HeadState]]) -> dict[int, Float32[Tensor, "heads key value"]]:
    """Average directions or states on CPU in float32."""
    totals: TensorMap = {}
    count = 0
    expected: set[int] | None = None
    for sample in samples:
        layers = set(sample)
        if expected is None:
            expected = layers
        elif layers != expected:
            raise ValueError("samples contain different decoder layers")
        for layer, tensor in sample.items():
            tensor = tensor.detach().float().cpu()
            if layer in totals and totals[layer].shape != tensor.shape:
                raise ValueError(f"shape mismatch at layer {layer}")
            totals[layer] = totals.get(layer, torch.zeros_like(tensor)) + tensor
        count += 1
    if count == 0:
        raise ValueError("cannot average zero samples")
    return {layer: tensor / count for layer, tensor in totals.items()}


def combine(
    directions: Iterable[dict[int, HeadState]], weights: Iterable[float] | None = None
) -> dict[int, Float32[Tensor, "heads key value"]]:
    """Weighted sum of directions. Missing weights are 1."""
    items = list(directions)
    if not items:
        raise ValueError("combine requires at least one direction")
    scales = [1.0] * len(items) if weights is None else [float(weight) for weight in weights]
    if len(scales) != len(items):
        raise ValueError("weights must match directions")
    layers = set(items[0])
    result: TensorMap = {}
    for layer in layers:
        shape = items[0][layer].shape
        total = torch.zeros(shape, dtype=torch.float32)
        for direction, weight in zip(items, scales, strict=True):
            if set(direction) != layers:
                raise ValueError("directions contain different decoder layers")
            tensor = direction[layer].detach().float().cpu()
            if tensor.shape != shape:
                raise ValueError(f"direction shape mismatch at layer {layer}")
            total = total + weight * tensor
        result[layer] = total
    return result


def truncate_rank(
    matrix: Float[Tensor, "*batch key value"], rank: int | None
) -> Float[Tensor, "*batch key value"]:
    """Return a rank-k SVD truncation. ``None`` and ``0`` keep the full matrix."""
    if not rank:
        return matrix
    if rank < 0:
        raise ValueError("rank must be non-negative")
    left: Float[Tensor, "*batch key spectrum"]
    singular: Float[Tensor, "*batch spectrum"]
    right: Float[Tensor, "*batch spectrum value"]
    left, singular, right = torch.linalg.svd(matrix.float(), full_matrices=False)
    return (left[..., :, :rank] * singular[..., None, :rank]) @ right[..., :rank, :]


def truncate_direction(
    direction: dict[int, HeadState], rank: int | None
) -> dict[int, Float32[Tensor, "heads key value"]]:
    """Truncate every layer matrix. The result stays on CPU in float32."""
    return {
        layer: truncate_rank(tensor, rank).detach().float().cpu()
        for layer, tensor in direction.items()
    }


def effective_rank(
    singular: Float[Tensor, "*batch spectrum"],
) -> Float[Tensor, "*batch"]:
    """Shannon effective rank. The last axis is the singular-value spectrum."""
    probability = singular / singular.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(singular.dtype).tiny
    )
    return (-torch.special.xlogy(probability, probability).sum(dim=-1)).exp()


def _row_scale(scale: Scale, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Scalar scale, or one value per batch row broadcast over a head matrix."""
    scales = torch.as_tensor(scale, device=device, dtype=dtype)
    if scales.ndim > 1 or (scales.ndim == 1 and scales.shape[0] not in (1, batch)):
        raise ValueError("scale must be a scalar or one value per batch row")
    if scales.ndim == 0:
        return scales
    return scales.reshape(-1, 1, 1, 1)


def add_delta(
    state: BatchState,
    delta: HeadState,
    scale: Scale,
    normalize: bool = True,
) -> BatchState:
    """Add ``scale * delta`` into ``state``.

    With ``normalize=True``, each head is rescaled so its Frobenius norm matches
    the current state. A zero state keeps the raw delta, because there is no
    norm to match.
    """
    if state.ndim < 3:
        raise ValueError("state must have a batch axis and a head matrix")
    correction = delta.to(device=state.device, dtype=state.dtype)
    if normalize:
        delta_norm: Float[Tensor, "heads 1 1"] = torch.linalg.matrix_norm(
            correction.float(), dim=(-2, -1), keepdim=True
        ).clamp_min(1e-8)
        state_norm: Float[Tensor, "batch heads 1 1"] = torch.linalg.matrix_norm(
            state.float(), dim=(-2, -1), keepdim=True
        )
        factor: Float[Tensor, "batch heads 1 1"] = torch.where(
            state_norm > 0, state_norm / delta_norm, torch.ones_like(state_norm)
        )
        correction = correction * factor.to(correction.dtype)
    correction = (
        _row_scale(scale, state.shape[0], device=state.device, dtype=state.dtype) * correction
    )
    state.add_(correction)
    return correction


def clamp_delta(
    state: BatchState,
    direction: HeadState,
    target_state: HeadState,
    scale: Scale,
) -> BatchState:
    """Move each head's projection on ``direction`` toward ``target_state``.

    ``scale`` is the fraction of the gap to close. The direction is normalized
    per head here, so callers can pass a raw target-minus-source matrix.
    """
    if state.ndim < 3:
        raise ValueError("state must have a batch axis and a head matrix")
    unit: Float32[Tensor, "heads key value"] = direction.detach().to(
        device=state.device, dtype=torch.float32
    )
    unit = unit / torch.linalg.matrix_norm(unit, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    target: Float32[Tensor, "heads key value"] = target_state.detach().to(
        device=state.device, dtype=torch.float32
    )
    before: Float[Tensor, "batch heads 1 1"] = (state.float() * unit).sum(
        dim=(-2, -1), keepdim=True
    )
    destination: Float[Tensor, "heads 1 1"] = (target * unit).sum(dim=(-2, -1), keepdim=True)
    gap: Float[Tensor, "batch heads 1 1"] = _row_scale(
        scale, state.shape[0], device=state.device, dtype=torch.float32
    ) * (destination - before)
    correction: BatchState = (gap * unit).to(dtype=state.dtype)
    state.add_(correction)
    return correction


class Metric(Protocol):
    """Observation computed while a direction is accumulated."""

    name: str

    def observe(
        self, target: BatchState, source: BatchState, mean_delta: HeadState
    ) -> Float[Tensor, "batch heads"]:
        """Return one value per batch row and head. Tensors are float CPU or CUDA."""


class FrobeniusDelta:
    """Frobenius norm of target minus source, per head."""

    name = "frobenius_delta"

    def observe(
        self, target: BatchState, source: BatchState, mean_delta: HeadState
    ) -> Float[Tensor, "batch heads"]:
        del mean_delta
        return torch.linalg.matrix_norm(target.float() - source.float(), dim=(-2, -1))


class EffectiveRank:
    """Effective rank of the target state, per head."""

    name = "effective_rank"

    def observe(
        self, target: BatchState, source: BatchState, mean_delta: HeadState
    ) -> Float[Tensor, "batch heads"]:
        del source, mean_delta
        return effective_rank(torch.linalg.svdvals(target.float()))


class CosineToMean:
    """Cosine between this pair's delta and the running mean delta."""

    name = "cosine_to_mean"

    def observe(
        self, target: BatchState, source: BatchState, mean_delta: HeadState
    ) -> Float[Tensor, "batch heads"]:
        delta: Float[Tensor, "batch heads channel"] = (target.float() - source.float()).flatten(
            start_dim=-2
        )
        mean: Float[Tensor, "heads channel"] = mean_delta.float().flatten(start_dim=-2)
        return torch.nn.functional.cosine_similarity(delta, mean.unsqueeze(0), dim=-1)


class Accumulator:
    """Online mean of target, source, and target-minus-source."""

    def __init__(self, shape: tuple[int, ...], metrics: Sequence[Metric] = ()) -> None:
        self.count = 0
        self.metrics = tuple(metrics)
        self.mean_delta: Float32[Tensor, "heads key value"] = torch.zeros(
            shape, dtype=torch.float32
        )
        self.mean_target: Float32[Tensor, "heads key value"] = torch.zeros(
            shape, dtype=torch.float32
        )
        self.mean_source: Float32[Tensor, "heads key value"] = torch.zeros(
            shape, dtype=torch.float32
        )
        self.observations: dict[str, list[Tensor]] = {metric.name: [] for metric in self.metrics}

    def update(self, target: BatchState, source: BatchState) -> None:
        """Update from a batch shaped ``[batch, heads, key, value]``."""
        target = target.detach().float().cpu()
        source = source.detach().float().cpu()
        if target.ndim != source.ndim or target.shape != source.shape:
            raise ValueError("target and source batches must have the same shape")
        if target.shape[1:] != self.mean_delta.shape:
            raise ValueError(
                f"expected [batch, *{tuple(self.mean_delta.shape)}], got {tuple(target.shape)}"
            )
        batch = target.shape[0]
        if batch == 0:
            raise ValueError("batch must not be empty")
        new_count = self.count + batch
        weight = batch / new_count
        self.mean_delta += weight * ((target - source).mean(0) - self.mean_delta)
        self.mean_target += weight * (target.mean(0) - self.mean_target)
        self.mean_source += weight * (source.mean(0) - self.mean_source)
        self.count = new_count
        for metric in self.metrics:
            value = metric.observe(target, source, self.mean_delta).detach().float().cpu()
            self.observations[metric.name].append(value)

    def summary(self) -> dict[str, Float32[Tensor, "batch heads"]]:
        """Concatenate metric observations along the batch axis."""
        return {
            name: torch.cat(values, dim=0) if values else torch.empty(0)
            for name, values in self.observations.items()
        }


def _same_layers(left: TensorMap, right: TensorMap) -> None:
    if set(left) != set(right):
        raise ValueError("states contain different decoder layers")
