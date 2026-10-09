"""Recurrent clamp used by the five-concept Qwen9B GDN cohort."""

import math
from typing import Any

import torch

from hybrid_steering.cache import (
    assert_nonrecurrent_unchanged as assert_nonrecurrent_unchanged,
)
from hybrid_steering.cache import (
    recurrent_tensor,
)
from hybrid_steering.cache import (
    snapshot_nonrecurrent as snapshot_nonrecurrent,
)


def rss_coefficients(directions, coefficients):
    """Scale only the features that actually have a direction at each layer."""
    layers = sorted(set().union(*(set(directions[name]) for name in coefficients)))
    result = {}
    for layer in layers:
        names = [name for name in coefficients if layer in directions[name]]
        pieces = [directions[name][layer] * float(coefficients[name]) for name in names]
        raw = sum(pieces[1:], pieces[0].clone())
        target = math.sqrt(sum(float(piece.square().sum()) for piece in pieces))
        scale = target / max(float(raw.norm()), 1e-12)
        result[layer] = {name: float(coefficients[name]) * scale for name in names}
    return result


def gram_inverse(flat_basis: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    gram = flat_basis @ flat_basis.T
    scale = torch.diag(gram).mean().clamp_min(1e-12)
    return torch.linalg.inv(gram + torch.eye(len(flat_basis), device=gram.device) * ridge * scale)


def make_clamp_runtime(
    cache: Any,
    directions: dict[str, dict[int, torch.Tensor]],
    deltas: dict[int, dict[str, float]],
    ridge: float = 1e-6,
) -> dict[int, dict[str, Any]]:
    """Project each recurrent state onto selected feature directions once."""
    result = {}
    for layer, by_feature in deltas.items():
        state = recurrent_tensor(cache, layer)
        names = tuple(by_feature)
        basis = torch.stack([directions[name][layer].to(state).float().flatten() for name in names])
        inverse = gram_inverse(basis, ridge)
        initial = inverse @ (basis @ state.float().flatten())
        result[layer] = {
            "names": names,
            "basis": basis,
            "inverse": inverse,
            "target": initial
            + torch.tensor([by_feature[name] for name in names], device=state.device),
        }
    return result


def make_runtime(
    cache: Any,
    directions: dict[str, dict[int, torch.Tensor]],
    active: list[str],
    alphas: dict[str, float],
) -> dict[int, dict[str, Any]]:
    if not active:
        return {}
    active_directions = {name: directions[name] for name in active}
    coefficients = {name: float(alphas[name]) for name in active}
    deltas = rss_coefficients(active_directions, coefficients)
    return make_clamp_runtime(cache, active_directions, deltas)


@torch.no_grad()
def apply_clamp(cache: Any, runtime: dict[int, dict[str, Any]], betas: dict[str, float]) -> None:
    """Apply one simultaneous recurrent correction from one state snapshot."""
    for layer, values in runtime.items():
        state = recurrent_tensor(cache, layer)
        current = values["inverse"] @ (values["basis"] @ state.float().flatten())
        correction = values["target"] - current
        scales = torch.tensor(
            [float(betas[name]) for name in values["names"]],
            device=state.device,
            dtype=torch.float32,
        )
        delta = (correction * scales) @ values["basis"]
        state.add_(delta.reshape(state.shape).to(state))


def finite_active(cache, runtime):
    for layer in runtime:
        if not torch.isfinite(recurrent_tensor(cache, layer)).all():
            raise FloatingPointError("nonfinite recurrent state after clamp")
