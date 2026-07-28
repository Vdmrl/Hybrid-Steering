from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file, save_file

from .cache import TensorMap
from .models import DirectionManifest


def save_direction(
    directory: str | Path,
    direction: TensorMap,
    manifest: DirectionManifest,
) -> None:
    """Save tensors and validated provenance as one directory artifact."""
    _validate_direction(direction, manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"layer_{layer_idx}": tensor.detach().float().cpu().contiguous()
        for layer_idx, tensor in direction.items()
    }
    save_file(tensors, directory / "direction.safetensors")
    (directory / "direction.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_direction(
    directory: str | Path,
) -> tuple[TensorMap, DirectionManifest]:
    directory = Path(directory)
    manifest = DirectionManifest.model_validate_json(
        (directory / "direction.json").read_text(encoding="utf-8")
    )
    tensors = load_file(directory / "direction.safetensors", device="cpu")
    direction: TensorMap = {}
    for key, tensor in tensors.items():
        prefix, separator, suffix = key.partition("_")
        if prefix != "layer" or separator != "_" or not suffix.isdigit():
            raise ValueError(f"invalid direction tensor key: {key}")
        direction[int(suffix)] = tensor.float()
    _validate_direction(direction, manifest)
    return direction, manifest


def _validate_direction(
    direction: TensorMap,
    manifest: DirectionManifest,
) -> None:
    indices = manifest.decoder_layer_indices_zero_based
    if set(direction) != set(indices):
        raise ValueError("direction layers do not match manifest")
    shapes = {layer_idx: list(tensor.shape) for layer_idx, tensor in direction.items()}
    if shapes != manifest.state_shapes:
        raise ValueError("direction tensor shapes do not match manifest")
