import torch
from torch import Tensor
from collections.abc import Mapping
from typing import Any, TypeVar, cast
from jaxtyping import Float, Float32

Value = TypeVar("Value")


def to_cpu(value: Value) -> Value:
    """Move tensors nested in common collections to the CPU."""
    if isinstance(value, torch.Tensor):
        return cast(Value, value.cpu())
    if isinstance(value, Mapping):
        return cast(Value, {key: to_cpu(item) for key, item in value.items()})
    if isinstance(value, list):
        return cast(Value, [to_cpu(item) for item in value])
    if isinstance(value, tuple):
        return cast(Value, tuple(to_cpu(item) for item in value))
    return value


def gdn_layers(model: Any) -> list[int]:
    """Return model layer indices containing GDN linear attention."""
    return [index for index, layer in enumerate(model.model.layers) if hasattr(layer, "linear_attn")]


def truncate_svd(matrix: Float32[Tensor, "*batch key value"], rank: int | None = 1) -> Float32[Tensor, "*batch key value"]:
    """Return the requested truncated-SVD approximation."""
    if not rank:
        return matrix
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[..., :, :rank] * singular[..., None, :rank]) @ vh[..., :rank, :]
