"""Post-prefill additive steering and coordinate clamp.

Both modes use the same target-minus-source direction. Clamp moves each head's
projection on that direction toward the mean target state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_steering import (
    DirectionManifest,
    Runner,
    add_delta,
    clamp_delta,
    collect_direction,
    gdn_layers,
    load_runtime,
    save_direction,
    truncate_direction,
)

PAIRS = (
    ("Это русский текст про дождь.", "This is English text about rain."),
    ("Завтра будет холодно.", "Tomorrow will be cold."),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "tiny"
        args.scale = 50.0
    model, tokenizer = load_runtime(args.model)
    layers = gdn_layers(model)
    runner = Runner(model, tokenizer, layers)
    collected = collect_direction(runner, PAIRS[:1] if args.smoke else PAIRS, batch_size=1)
    direction = truncate_direction(collected.delta, args.rank or None)
    manifest = DirectionManifest(
        model_id=args.model,
        target="russian",
        source="english",
        example_ids=["pair-0"],
        decoder_layer_indices=sorted(direction),
        state_shapes={layer: list(tensor.shape) for layer, tensor in direction.items()},
        rank=args.rank or None,
    )
    save_direction(
        args.output / "direction",
        direction,
        manifest,
        mean_target=collected.mean_target,
        mean_source=collected.mean_source,
    )
    prompt = ["Describe the weather."]
    device = next(model.parameters()).device
    stats = {}
    decode_logits = {}
    with torch.inference_mode():
        for mode in ("none", "add", "clamp"):
            trace_runner = Runner(model, tokenizer, layers)
            inputs, mask = trace_runner._inputs(prompt)
            cache, baseline_logits, mask = trace_runner.prefill(inputs, mask, None, 0.0)
            for layer in layers:
                state = cache.layers[layer].recurrent_states[0]
                delta = direction[layer].to(device)
                target_state = collected.mean_target[layer].to(device)
                if mode == "add":
                    add_delta(state, delta, args.scale, normalize=False)
                elif mode == "clamp":
                    clamp_delta(state, delta, target_state, args.scale)
            steered = model(
                input_ids=baseline_logits.argmax(-1)[:, None],
                attention_mask=torch.cat((mask, torch.ones_like(mask[:, :1])), 1),
                past_key_values=cache,
                use_cache=True,
            )
            decode_logits[mode] = steered.logits[:, -1]
            stats[mode] = {
                "logit_delta_vs_baseline": float(
                    (decode_logits[mode] - decode_logits.get("none", decode_logits[mode]))
                    .abs()
                    .mean()
                ),
                "finite": bool(torch.isfinite(decode_logits[mode]).all()),
            }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "clamp.json").write_text(json.dumps(stats, indent=2) + "\n")
    if not stats["add"]["finite"] or not stats["clamp"]["finite"]:
        raise SystemExit("clamp or add produced non-finite logits")
    if (
        stats["add"]["logit_delta_vs_baseline"] == 0
        or stats["clamp"]["logit_delta_vs_baseline"] == 0
    ):
        raise SystemExit("intervention did not change the next-token logits")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
