"""Select a residual layer from a complete, matched concept-metric grid."""

import math


def select_layer(cells, layers, strengths, relative_threshold=0.95):
    if not 0 < relative_threshold <= 1:
        raise ValueError("relative threshold must be in (0, 1]")
    if len(set(layers)) != len(layers) or len(set(strengths)) != len(strengths):
        raise ValueError("duplicate layers or strengths")
    expected = {(layer, float(scale)) for layer in layers for scale in strengths}
    actual = [(c["layer"], float(c["strength"])) for c in cells]
    if not expected or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("need the complete unique layer/strength grid")
    if any(not math.isfinite(c["value"]) or c["value"] < 0 for c in cells):
        raise ValueError("metric must be finite and nonnegative")
    maximum = max(c["value"] for c in cells)
    if maximum == 0:
        return {"status": "no_expression", "layer": None, "maximum": 0}
    threshold = relative_threshold * maximum
    peaks = {layer: max(c["value"] for c in cells if c["layer"] == layer) for layer in layers}
    eligible = [c for c in cells if c["value"] >= threshold]
    winner = min(
        eligible, key=lambda c: (c["strength"], -c["value"], -peaks[c["layer"]], c["layer"])
    )
    return {
        "status": "selected",
        "layer": winner["layer"],
        "strength": winner["strength"],
        "value": winner["value"],
        "maximum": maximum,
        "threshold": threshold,
        "relative_threshold": relative_threshold,
    }
