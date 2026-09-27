"""Record final GDN state norms for texts of different lengths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from hybrid_steering import Runner, final_states, gdn_layers, load_runtime

TEXTS = (
    "Rain.",
    "Rain on the window for a whole quiet hour.",
    "Rain on the window while the street stays empty and the bus is late again tonight.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "tiny"
    model, tokenizer = load_runtime(args.model)
    layers = gdn_layers(model)
    runner = Runner(model, tokenizer, layers)
    states = final_states(runner, list(TEXTS))
    rows = []
    for index, text in enumerate(TEXTS):
        for layer, tensor in states.items():
            rows.append(
                {
                    "text": text,
                    "tokens": len(tokenizer.encode(text)),
                    "layer": layer,
                    "frobenius": float(
                        torch.linalg.matrix_norm(tensor[index].float(), dim=(-2, -1)).mean()
                    ),
                }
            )
    if not rows or any(math.isnan(row["frobenius"]) for row in rows):
        raise SystemExit("state dynamics produced no finite norms")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "norms.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} norms", flush=True)


if __name__ == "__main__":
    main()
