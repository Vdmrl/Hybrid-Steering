"""Small, shared helpers for frozen experiment plans."""

import json
from pathlib import Path

from .steering import SteeringConfig


def conditions(grid: dict) -> list[dict]:
    cases = [{"method": "baseline", "strength": 0}]
    cases += [{"method": "residual", "layer": layer, "strength": strength}
              for layer in grid["residual_layers"] for strength in grid["residual_strengths"]]
    cases += [{"method": "gdn_rank5", "strength": strength, "normalize_to_full": True}
              for strength in grid["gdn_rank5_normalized_strengths"]]
    cases += [{"method": "gdn_clamp_rank1", "strength": strength, "normalize_to_full": True}
              for strength in grid["gdn_rank1_clamp_normalized_strengths"]]
    checked = [SteeringConfig(**row) for row in cases]
    if len(checked) != len(set(checked)) or len(checked) != 26:
        raise ValueError("Campaign must have 26 unique conditions including baseline")
    return cases


def write_unchanged(path: Path, payload: dict) -> None:
    """Never silently edit the scientific plan of an existing run."""
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Existing plan changed; use a new campaign version: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
