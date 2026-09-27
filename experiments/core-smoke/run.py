"""Check that a cache intervention changes only the recurrent state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_steering import (
    apply_direction,
    assert_nonrecurrent_unchanged,
    extract_recurrent,
    gdn_layers,
    load_runtime,
    snapshot_nonrecurrent,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "tiny"
    model, tokenizer = load_runtime(args.model)
    device = next(model.parameters()).device
    encoded = tokenizer(["A short prompt."], add_special_tokens=False, return_tensors="pt").to(
        device
    )
    with torch.inference_mode():
        cache = model(**encoded, use_cache=True).past_key_values
        before = snapshot_nonrecurrent(cache)
        layers = gdn_layers(model)
        direction = {
            layer: torch.ones_like(cache.layers[layer].recurrent_states[0][0]).cpu()
            for layer in layers
        }
        base = extract_recurrent(cache)
        apply_direction(cache, direction, args.scale, normalize=False)
        assert_nonrecurrent_unchanged(before, cache)
        after = extract_recurrent(cache)
    delta = float((after[layers[0]] - base[layers[0]]).abs().mean())
    if delta <= 0:
        raise SystemExit("recurrent state did not change")
    payload = {"scale": args.scale, "mean_abs_delta": delta, "layers": layers}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "smoke.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
