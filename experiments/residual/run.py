"""Residual-stream baseline. This is not a GDN intervention.

The direction is the last-token residual of the target text minus the source
text, added on prompt tokens only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_steering import load_runtime
from hybrid_steering.direction import concept_sides, load_concept_pairs, target_and_source
from hybrid_steering.judge import score_rows


def last_hidden(model, tokenizer, text: str) -> torch.Tensor:
    encoded = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(
        next(model.parameters()).device
    )
    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True, use_cache=False)
    return output.hidden_states[-1][0, -1].float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concept", default="en-ru")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    source_name, target_name = concept_sides(args.concept, None, None)
    model, tokenizer = load_runtime(args.model)
    target, source = target_and_source(load_concept_pairs(args.concept)[0])
    direction = last_hidden(model, tokenizer, target) - last_hidden(model, tokenizer, source)
    direction = direction / direction.norm().clamp_min(1e-8)
    seen = {"calls": 0}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > 1:
            hidden.add_(args.scale * direction.to(device=hidden.device, dtype=hidden.dtype))
            seen["calls"] += 1
        return output

    handles = [layer.register_forward_hook(hook) for layer in model.model.layers]
    try:
        encoded = tokenizer(
            ["What might happen if someone misses the last bus home?"],
            add_special_tokens=False,
            return_tensors="pt",
        )
        encoded = encoded.to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
    finally:
        for handle in handles:
            handle.remove()
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    payload = {
        "hook_calls": seen["calls"],
        "direction_norm": 1.0,
        "prompt": "What might happen if someone misses the last bus home?",
        "response": text,
        "target": target_name,
        "source": source_name,
    }
    score_rows([payload], target_name)
    if seen["calls"] < 1:
        raise SystemExit("residual hook did not run on the prompt")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "residual.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
