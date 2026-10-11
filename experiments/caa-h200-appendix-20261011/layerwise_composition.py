"""RSS coefficient scaling for features routed to different recurrent layers."""

from __future__ import annotations

import math


def rss_coefficients_by_layer(directions, coefficients):
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
