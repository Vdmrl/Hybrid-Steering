"""Residual-stream baseline. This is not a GDN intervention.

The direction is the last-token residual of the target text minus the source
text, added on prompt tokens only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from hybrid_steering import load_runtime


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
    model, tokenizer = load_runtime(args.model)
    slug = args.concept.replace("->", "-")
    path = Path(
        hf_hub_download(
            "hybrid-steering/hybrid-steering-concepts",
            f"concepts/{slug}/data/pairs.jsonl",
            repo_type="dataset",
        )
    )
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    target, source = row["positive_text"], row["negative_text"]
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
    payload = {"hook_calls": seen["calls"], "direction_norm": 1.0, "response": text}
    if seen["calls"] < 1:
        raise SystemExit("residual hook did not run on the prompt")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "residual.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
