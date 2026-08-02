"""Small RSS composition and recurrent-state clamp helpers for this experiment."""

from __future__ import annotations

import math
from typing import Any

import torch
from hybrid_steering.cache import recurrent_tensor


def rss_coefficients(
    directions: dict[str, dict[int, torch.Tensor]], coefficients: dict[str, float]
) -> dict[int, dict[str, float]]:
    """Rescale an additive composition to its root-sum-square norm per layer."""
    names = tuple(coefficients)
    result = {}
    for layer in directions[names[0]]:
        pieces = [directions[name][layer] * coefficients[name] for name in names]
        raw = sum(pieces[1:], pieces[0].clone())
        target = math.sqrt(sum(float(piece.square().sum()) for piece in pieces))
        scale = target / max(float(raw.norm()), 1e-12)
        result[layer] = {name: coefficients[name] * scale for name in names}
    return result


def gram_inverse(flat_basis: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    gram = flat_basis @ flat_basis.T
    scale = torch.diag(gram).mean().clamp_min(1e-12)
    return torch.linalg.inv(
        gram + torch.eye(len(flat_basis), device=gram.device) * ridge * scale
    )


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
        basis = torch.stack(
            [directions[name][layer].to(state).float().flatten() for name in names]
        )
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


@torch.no_grad()
def clamp_cache(cache: Any, runtime: dict[int, dict[str, Any]], beta: float) -> None:
    for layer, values in runtime.items():
        state = recurrent_tensor(cache, layer)
        current = values["inverse"] @ (values["basis"] @ state.float().flatten())
        correction = values["target"] - current
        state.add_(
            (correction @ values["basis"]).reshape(state.shape).to(state), alpha=beta
        )
