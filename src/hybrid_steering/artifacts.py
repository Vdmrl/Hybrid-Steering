"""Save and load a direction directory.

The directory holds ``direction.safetensors`` and ``direction.json``. Optional
mean states use the prefixes ``mean_target_`` and ``mean_source_``. A legacy
``deltas.pt`` file is still readable: its ``concept_b - concept_a`` delta is
the same target-minus-source direction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .models import DirectionManifest
from .state import TensorMap


def save_direction(
    directory: str | Path,
    direction: TensorMap,
    manifest: DirectionManifest,
    *,
    mean_target: TensorMap | None = None,
    mean_source: TensorMap | None = None,
) -> None:
    """Write tensors and a short label file as one directory."""
    _validate(direction, manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = _prefixed("layer_", direction)
    if mean_target is not None:
        _same_layers(mean_target, direction, "mean_target")
        tensors.update(_prefixed("mean_target_", mean_target))
    if mean_source is not None:
        _same_layers(mean_source, direction, "mean_source")
        tensors.update(_prefixed("mean_source_", mean_source))
    save_file(tensors, directory / "direction.safetensors")
    (directory / "direction.json").write_text(_manifest_json(manifest), encoding="utf-8")


def load_direction(
    path: str | Path,
) -> tuple[TensorMap, DirectionManifest, TensorMap | None, TensorMap | None]:
    """Load a directory artifact or a legacy ``deltas.pt`` file.

    Returns direction, manifest, mean target, and mean source. Legacy files
    have no mean states.
    """
    path = Path(path)
    if path.is_dir():
        return _load_directory(path)
    if path.suffix == ".pt":
        return _load_legacy_pt(path)
    raise ValueError(f"direction artifact must be a directory or a .pt file, got {path}")


def _load_directory(
    directory: Path,
) -> tuple[TensorMap, DirectionManifest, TensorMap | None, TensorMap | None]:
    manifest = _manifest_from_json((directory / "direction.json").read_text(encoding="utf-8"))
    tensors = load_file(directory / "direction.safetensors", device="cpu")
    direction = _take(tensors, "layer_")
    mean_target = _take(tensors, "mean_target_") or None
    mean_source = _take(tensors, "mean_source_") or None
    _validate(direction, manifest)
    return direction, manifest, mean_target, mean_source


def _load_legacy_pt(
    path: Path,
) -> tuple[TensorMap, DirectionManifest, TensorMap | None, TensorMap | None]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    deltas = artifact.get("deltas")
    if not isinstance(deltas, dict) or not deltas:
        raise ValueError("legacy artifact must contain non-empty deltas")
    target = artifact.get("concept_b") or artifact.get("target")
    source = artifact.get("concept_a") or artifact.get("source")
    if not isinstance(target, str) or not isinstance(source, str):
        raise ValueError("legacy artifact must name the target and source concepts")
    formula = artifact.get("delta")
    if formula is not None and formula != f"{target} - {source}":
        raise ValueError("legacy delta must be target minus source")
    direction = {int(layer): tensor.detach().float().cpu() for layer, tensor in deltas.items()}
    manifest = DirectionManifest(
        model_id=str(artifact.get("model") or "unknown"),
        target=target,
        source=source,
        example_ids=[f"legacy-{index}" for index in range(int(artifact.get("pairs") or 1))],
        decoder_layer_indices=sorted(direction),
        state_shapes={layer: list(tensor.shape) for layer, tensor in direction.items()},
    )
    return direction, manifest, None, None


def _prefixed(prefix: str, values: TensorMap) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}{layer}": tensor.detach().float().cpu().contiguous()
        for layer, tensor in values.items()
    }


def _take(tensors: dict[str, torch.Tensor], prefix: str) -> TensorMap:
    result: TensorMap = {}
    for key, tensor in tensors.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if not suffix.isdigit():
            raise ValueError(f"invalid direction tensor key: {key}")
        result[int(suffix)] = tensor.float()
    return result


def _same_layers(values: TensorMap, direction: TensorMap, label: str) -> None:
    if set(values) != set(direction):
        raise ValueError(f"{label} layers do not match the direction")


def _manifest_json(manifest: DirectionManifest) -> str:
    payload = asdict(manifest)
    payload["state_shapes"] = {str(layer): shape for layer, shape in manifest.state_shapes.items()}
    return json.dumps(payload, indent=2) + "\n"


def _manifest_from_json(text: str) -> DirectionManifest:
    raw = json.loads(text)
    known = {item.name for item in fields(DirectionManifest)}
    payload = {key: value for key, value in raw.items() if key in known}
    payload["state_shapes"] = {
        int(layer): shape for layer, shape in payload["state_shapes"].items()
    }
    return DirectionManifest(**payload)


def _validate(direction: TensorMap, manifest: DirectionManifest) -> None:
    if set(direction) != set(manifest.decoder_layer_indices):
        raise ValueError("direction layers do not match manifest")
    shapes = {layer: list(tensor.shape) for layer, tensor in direction.items()}
    if shapes != manifest.state_shapes:
        raise ValueError("direction tensor shapes do not match manifest")
