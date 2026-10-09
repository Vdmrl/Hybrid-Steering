"""Configure a reusable model for one explicit steering condition.

No hooks remain installed between calls. Generation delegates to the existing
decoder and intervention math in concept_comparison.py and run.py.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from hybrid_steering.extract import sha256
from hybrid_steering.architecture import decoder_layers, recurrent_family


GDN_ADDITIVE = {"gdn_full", "gdn_rank1", "gdn_rank5"}
GDN_CLAMP = {"gdn_clamp_rank1", "gdn_clamp_rank5"}
METHODS = {"baseline", "residual", "residual_clamp", "mamba_full"} | GDN_ADDITIVE | GDN_CLAMP


@dataclass(frozen=True)
class SteeringConfig:
    method: str
    strength: float
    layer: int = -1
    normalize_to_full: bool = False

    def __post_init__(self) -> None:
        ranked = re.fullmatch(r"(?:gdn|mamba)_(?:clamp_)?rank([1-9][0-9]*)", self.method)
        if (self.method not in METHODS and not ranked) or type(self.strength) not in (int, float):
            raise ValueError("Unknown steering method or nonnumeric strength")
        if type(self.layer) is not int or not math.isfinite(self.strength):
            raise ValueError("Layer must be an integer and strength must be finite")
        if type(self.normalize_to_full) is not bool:
            raise ValueError("normalize_to_full must be boolean")
        if self.method == "baseline":
            if self.strength != 0 or self.layer != -1 or self.normalize_to_full:
                raise ValueError("Baseline must be a true zero-strength no-op")
        elif self.strength <= 0:
            raise ValueError("Steering strength must be positive")
        if self.method in {"residual", "residual_clamp"} and self.layer < 0:
            raise ValueError("Residual steering requires a nonnegative layer")
        if self.method not in {"residual", "residual_clamp"} and self.layer != -1:
            raise ValueError("GDN steering applies to all recurrent layers")
        if self.normalize_to_full and not ranked:
            raise ValueError("Only truncated GDN directions can be normalized to full-rank")

    @property
    def schedule(self) -> str:
        if self.method == "baseline":
            return "none"
        if self.method in {"residual", "residual_clamp"}:
            return "every_token"
        return "clamp_every_token" if self.method.startswith(("gdn_clamp_rank", "mamba_clamp_rank")) else "once"


@dataclass
class Directions:
    artifact: dict
    sha256: dict[str, str]
    concept: str | None = None
    model: str | None = None
    model_revision: str | None = None
    prepared: dict = field(default_factory=dict, repr=False)


def load_directions(path: Path) -> Directions:
    """Load trusted .pt artifacts from this project, not arbitrary downloads.

    Accept a new extraction's directions.pt or an existing concept directory
    containing gdn.pt and optionally residual.pt. Ignore unrelated metadata
    fields that can occur in both historical source files.
    """
    import torch

    path = path.resolve()
    combined = path / "directions.pt" if path.is_dir() else path
    provenance = []
    if combined.is_file():
        files = [combined]
        artifact = torch.load(combined, map_location="cpu", weights_only=False)
        manifest = combined.parent / "source-manifest.json"
        if manifest.exists():
            source = json.loads(manifest.read_text())
            if source.get("directions_sha256") != sha256(combined):
                raise ValueError(f"Incomplete or changed direction source: {manifest}")
            provenance = [source]
    elif path.is_dir() and (path / "gdn.pt").is_file():
        files = [path / "gdn.pt"]
        gdn = torch.load(files[0], map_location="cpu", weights_only=False)
        if not isinstance(gdn, dict):
            raise ValueError("GDN direction file must contain a dictionary")
        artifact = {key: gdn[key] for key in ("gdn_full", "gdn_rank1", "gdn_rank5", "clamp_factors")
                    if key in gdn}
        residual = path / "residual.pt"
        if residual.is_file():
            files.append(residual)
            saved = torch.load(residual, map_location="cpu", weights_only=False)
            if not isinstance(saved, dict):
                raise ValueError("Residual direction file must contain a dictionary")
            artifact.update({key: saved[key] for key in ("units", "targets", "neutral_targets", "residual_directions")
                             if key in saved})
        for name in ("gdn-source-manifest.json", "residual-source-manifest.json"):
            manifest = path.parent / "provenance" / name
            if manifest.exists():
                provenance.append(json.loads(manifest.read_text()))
    else:
        raise FileNotFoundError(f"No direction artifact at {path}")
    if not isinstance(artifact, dict) or not ("gdn_full" in artifact or "residual_directions" in artifact):
        raise ValueError("Artifact contains neither GDN nor residual directions")
    concepts = {source.get("identity", {}).get("concept") or
                source.get("identity", {}).get("config", {}).get("concept")
                for source in provenance} - {None}
    models = {source.get("identity", {}).get("model") or
              source.get("identity", {}).get("config", {}).get("model")
              for source in provenance} - {None}
    revisions = {source.get("identity", {}).get("model_revision") or source.get("model_revision")
                 for source in provenance} - {None}
    if any(len(values) > 1 for values in (concepts, models, revisions)):
        raise ValueError("GDN and residual source manifests disagree")
    return Directions(artifact, {file.name: sha256(file) for file in files},
                      next(iter(concepts), None), next(iter(models), None),
                      next(iter(revisions), None))


class SteeredModel:
    """Model + tokenizer + directions + one immutable steering configuration."""

    def __init__(self, model, tokenizer, directions: Directions, config: SteeringConfig):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.directions = directions
        self.artifact = dict(directions.artifact)
        if config.method.startswith(("gdn_", "mamba_")):
            family = recurrent_family(model)
            if not config.method.startswith(family + "_") or self.artifact.get("state_family", "gdn") != family:
                raise ValueError("Steering method, model and direction state family must agree")
        if config.method in {"residual", "residual_clamp"}:
            if any(config.layer not in self.artifact.get(name, {})
                   for name in ("units", "targets", "neutral_targets")):
                raise ValueError(f"No residual direction for layer {config.layer}")
        elif config.method.startswith(("gdn_", "mamba_")):
            self._configure_gdn()

    def _configure_gdn(self) -> None:
        from hybrid_steering.state import analyze, match_head_norm, reconstruct

        config = self.config
        full = self.artifact.get("gdn_full")
        if not full:
            raise ValueError("GDN steering requires gdn_full in the artifact")
        if config.method == "mamba_full":
            self.artifact["mamba_full"] = full
            return
        if config.method == "gdn_full":
            return
        rank = int(config.method.rsplit("rank", 1)[1])
        key = rank, config.normalize_to_full
        prepared = self.directions.prepared.setdefault(key, {})
        if "direction" not in prepared:
            direction = self.artifact.get(f"gdn_rank{rank}")
            if direction is None:
                _, factors = analyze(full, "cpu", rank=rank)
                direction = reconstruct(factors)
            if config.normalize_to_full:
                direction, report = match_head_norm(direction, full)
                if any(item["capped_heads"] for item in report.values()):
                    raise ValueError("Per-head normalization hit its gain cap")
            prepared["direction"] = direction
        if config.method.startswith(("gdn_rank", "mamba_rank")):
            self.artifact[f"gdn_rank{rank}"] = prepared["direction"]
            if config.method.startswith("mamba_rank"):
                self.artifact[config.method] = prepared["direction"]
            return
        if "factors" not in prepared:
            _, prepared["factors"] = analyze(prepared["direction"], "cpu", rank=rank)
        factors = {}
        for layer, values in prepared["factors"].items():
            parameter = next(decoder_layers(self.model)[layer].parameters())
            factors[layer] = tuple(value.to(device=parameter.device, dtype=parameter.dtype)
                                   for value in values)
        self.artifact["clamp_factors"] = factors

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Return one answer per prompt; no-thinking, greedy, existing decoder."""
        if max_new_tokens < 1 or any(not isinstance(p, str) or not p for p in prompts):
            raise ValueError("Prompts must be nonempty strings; token limit must be positive")
        if not prompts:
            return []
        if self.config.method.startswith("mamba_") or (self.config.method.startswith("gdn_clamp_rank") and self.config.method != "gdn_clamp_rank1"):
            from hybrid_steering.runner import generate_batch
            clamped = self.config.method.startswith("mamba_clamp_rank") or self.config.method.startswith("gdn_clamp_rank")
            return generate_batch(self.model, self.tokenizer, prompts,
                                  None if clamped else self.artifact[self.config.method],
                                  self.artifact.get("clamp_factors") if clamped else None,
                                  self.config.strength, "clamp" if clamped else "once", "cuda", max_new_tokens,
                                  system=None, enable_thinking=False)
        from hybrid_steering.comparison import generate_condition
        rows = [{"prompt": prompt} for prompt in prompts]
        return generate_condition(self.model, self.tokenizer, rows, self.artifact,
                                  self.config.method, self.config.layer,
                                  self.config.strength, max_new_tokens)


def steer_model(model, tokenizer, directions_path: Path, config: SteeringConfig) -> SteeredModel:
    """Return a ready-to-generate model; rank-k is derived from the saved full direction."""
    return SteeredModel(model, tokenizer, load_directions(directions_path), config)
