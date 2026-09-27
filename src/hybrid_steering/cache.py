"""Cache helpers for absolute decoder-layer recurrent state."""

import copy
import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .state import BatchState, Scale, TensorMap, add_delta


def gdn_layers(model: Any) -> list[int]:
    """Absolute decoder indices whose block owns ``linear_attn``."""
    return [
        index for index, layer in enumerate(model.model.layers) if hasattr(layer, "linear_attn")
    ]


def gdn_layer_indices(cache: Any) -> list[int]:
    """Absolute decoder indices whose cache layer holds a recurrent state."""
    return [
        layer
        for layer, block in enumerate(_layers(cache))
        if list(_as_tensors(getattr(block, "recurrent_states", None)))
    ]


def extract_recurrent(cache: Any, *, device: str | torch.device = "cpu") -> dict[int, BatchState]:
    """Copy one recurrent matrix from every GDN layer as float32."""
    result: TensorMap = {}
    for layer, block in enumerate(_layers(cache)):
        tensors = list(_as_tensors(getattr(block, "recurrent_states", None)))
        if not tensors:
            continue
        if len(tensors) != 1:
            raise RuntimeError(
                f"expected one recurrent tensor at layer {layer}, got {len(tensors)}"
            )
        result[layer] = tensors[0].detach().to(device=device, dtype=torch.float32).clone()
    if not result:
        raise RuntimeError("no recurrent states found in cache")
    return result


def recurrent_tensor(cache: Any, layer: int) -> BatchState:
    """Return the live recurrent tensor at an absolute decoder index."""
    layers = _layers(cache)
    if layer < 0 or layer >= len(layers):
        raise IndexError(f"decoder layer index out of range: {layer}")
    tensors = list(_as_tensors(getattr(layers[layer], "recurrent_states", None)))
    if len(tensors) != 1:
        raise RuntimeError(f"expected one recurrent tensor at layer {layer}, got {len(tensors)}")
    return tensors[0]


def apply_direction(
    cache: Any,
    direction: TensorMap,
    scale: Scale,
    *,
    layers: Iterable[int] | None = None,
    normalize: bool = False,
) -> None:
    """Add a direction into selected recurrent states. KV and convolution stay put."""
    for layer in _selected(direction, layers):
        add_delta(recurrent_tensor(cache, layer), direction[layer], scale, normalize)


def replace_state(
    cache: Any,
    donor: TensorMap,
    *,
    layers: Iterable[int] | None = None,
) -> None:
    """Copy donor recurrent states over the cache without touching other tensors."""
    for layer in _selected(donor, layers):
        target = recurrent_tensor(cache, layer)
        source = donor[layer].to(device=target.device, dtype=target.dtype)
        if target.shape != source.shape:
            raise ValueError(f"donor shape mismatch at layer {layer}")
        target.copy_(source)


def clone_cache(cache: Any) -> Any:
    """Deep-copy a cache and reject shared tensor storage."""
    cloned = copy.deepcopy(cache)
    original = list(_walk_tensors(cache))
    copied = list(_walk_tensors(cloned))
    if [path for path, _ in original] != [path for path, _ in copied]:
        raise RuntimeError("cache deepcopy changed the tensor inventory")
    for (path, source), (_, target) in zip(original, copied, strict=True):
        if source.untyped_storage().data_ptr() == target.untyped_storage().data_ptr():
            raise RuntimeError(f"cache deepcopy shares tensor storage at {path}")
    return cloned


def snapshot_nonrecurrent(cache: Any) -> dict[str, str]:
    """Hash every cache tensor that is not a recurrent state."""
    return {
        path: _tensor_digest(tensor)
        for path, tensor in _walk_tensors(cache)
        if "recurrent_states" not in path
    }


def assert_nonrecurrent_unchanged(before: Mapping[str, str], cache: Any) -> None:
    """Raise when KV, convolution, or any other non-recurrent tensor changed."""
    after = snapshot_nonrecurrent(cache)
    if dict(before) == after:
        return
    changed = [
        path for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path)
    ]
    raise AssertionError(
        "non-recurrent cache changed during intervention: " + ", ".join(changed[:20])
    )


def _selected(values: TensorMap, layers: Iterable[int] | None) -> tuple[int, ...]:
    selected = tuple(values) if layers is None else tuple(layers)
    if len(selected) != len(set(selected)):
        raise ValueError("decoder layer indices must be unique")
    for layer in selected:
        if layer < 0:
            raise ValueError("decoder layer indices are zero-based and non-negative")
        if layer not in values:
            raise KeyError(f"no tensor for decoder layer {layer}")
    return selected


def _layers(cache: Any) -> Any:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("cache must expose a layers sequence")
    return layers


def _as_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _as_tensors(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _as_tensors(item)


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _walk_tensors(
    obj: Any,
    path: str = "cache",
    seen: set[int] | None = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, Mapping):
        for key in sorted(obj, key=str):
            yield from _walk_tensors(obj[key], f"{path}[{key!r}]", seen)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from _walk_tensors(value, f"{path}[{index}]", seen)
    elif hasattr(obj, "__dict__"):
        for name in sorted(vars(obj)):
            yield from _walk_tensors(getattr(obj, name), f"{path}.{name}", seen)
