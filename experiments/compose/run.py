"""Sum two concept directions and check the raw intervention stays linear.

Normalization is intentionally off. State-norm matching depends on the current
state, so the sum of two normalized interventions is a different operation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_steering import Runner, add_delta, combine, gdn_layers, load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "tiny"
    torch.manual_seed(0)
    model, tokenizer = load_runtime(args.model)
    layers = gdn_layers(model)
    device = next(model.parameters()).device
    first = {}
    second = {}
    for layer in layers:
        module = model.model.layers[layer].linear_attn
        shape = (module.num_v_heads, module.head_k_dim, module.head_v_dim)
        first[layer] = torch.randn(shape)
        second[layer] = torch.randn(shape)
    summed = combine([first, second])
    math_error = max(
        float((summed[layer] - first[layer] - second[layer]).abs().max()) for layer in layers
    )
    runner = Runner(model, tokenizer, layers)
    inputs, mask = runner._inputs(["Describe a quiet morning."])
    cache, _logits, _mask = runner.prefill(inputs, mask, None, 0.0)
    layer = layers[0]
    state = cache.layers[layer].recurrent_states[0]
    separate = state.clone()
    together = state.clone()
    add_delta(separate, first[layer].to(device), 1.0, normalize=False)
    add_delta(separate, second[layer].to(device), 1.0, normalize=False)
    add_delta(together, summed[layer].to(device), 1.0, normalize=False)
    apply_error = float((separate - together).abs().max())
    if math_error > 1e-5 or apply_error > 1e-4:
        raise SystemExit(f"composition failed: math={math_error} apply={apply_error}")
    payload = {"math_error": math_error, "apply_error": apply_error, "layers": layers}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "compose.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
