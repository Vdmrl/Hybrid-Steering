from __future__ import annotations

import copy
import hashlib
import io
from collections.abc import Iterable, Mapping
from typing import Any

import torch

TensorMap = dict[int, torch.Tensor]


def gdn_layer_indices(cache: Any) -> list[int]:
    """Return absolute zero-based decoder indices containing recurrent state."""
    return [
        layer_idx
        for layer_idx, layer in enumerate(_layers(cache))
        if list(_as_tensors(getattr(layer, "recurrent_states", None)))
    ]


def extract_recurrent(cache: Any, *, device: str | torch.device = "cpu") -> TensorMap:
    """Copy one recurrent matrix from every GDN layer as FP32."""
    result: TensorMap = {}
    for layer_idx, layer in enumerate(_layers(cache)):
        tensors = list(_as_tensors(getattr(layer, "recurrent_states", None)))
        if not tensors:
            continue
        if len(tensors) != 1:
            raise RuntimeError(
                f"expected one recurrent tensor at layer {layer_idx}, "
                f"got {len(tensors)}"
            )
        result[layer_idx] = (
            tensors[0].detach().to(device=device, dtype=torch.float32).clone()
        )
    if not result:
        raise RuntimeError("no recurrent states found in cache")
    return result


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
    """Hash KV, convolution, and other cache tensors."""
    return {
        path: _tensor_digest(tensor)
        for path, tensor in _walk_tensors(cache)
        if "recurrent_states" not in path
    }


def assert_nonrecurrent_unchanged(before: Mapping[str, str], cache: Any) -> None:
    after = snapshot_nonrecurrent(cache)
    if dict(before) == after:
        return
    paths = sorted(set(before) | set(after))
    changed = [path for path in paths if before.get(path) != after.get(path)]
    raise AssertionError(
        "non-recurrent cache changed during intervention: " + ", ".join(changed[:20])
    )


def recurrent_tensor(cache: Any, layer_idx: int) -> torch.Tensor:
    layers = _layers(cache)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"decoder layer index out of range: {layer_idx}")
    tensors = list(_as_tensors(getattr(layers[layer_idx], "recurrent_states", None)))
    if len(tensors) != 1:
        raise RuntimeError(
            f"expected one recurrent tensor at layer {layer_idx}, got {len(tensors)}"
        )
    return tensors[0]


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
    buffer = io.BytesIO()
    torch.save(tensor.detach().contiguous().cpu(), buffer)
    return hashlib.sha256(buffer.getbuffer()).hexdigest()


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
