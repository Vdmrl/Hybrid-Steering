"""Record final GDN state norms for texts of different lengths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset

from hybrid_steering import Runner, final_states, gdn_layers, load_runtime

DATASET = "claran/pg19-sample"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--length", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model, tokenizer = load_runtime(args.model)
    texts = []
    for row in load_dataset(args.dataset, split="train", streaming=True).shuffle(
        seed=args.seed, buffer_size=1_000
    ):
        ids = tokenizer(
            row["text"], add_special_tokens=False, truncation=True, max_length=args.length
        ).input_ids
        if len(ids) < args.length:
            continue
        texts.append(tokenizer.decode(ids[: args.length], clean_up_tokenization_spaces=False))
        if len(texts) == args.count:
            break
    if len(texts) < args.count:
        raise SystemExit(f"found only {len(texts)} texts of {args.length} tokens")
    layers = gdn_layers(model)
    runner = Runner(model, tokenizer, layers)
    rows = []
    for index, text in enumerate(texts):
        states = final_states(runner, [text])
        for layer, tensor in states.items():
            rows.append(
                {
                    "text_id": index,
                    "tokens": args.length,
                    "layer": layer,
                    "frobenius": float(
                        torch.linalg.matrix_norm(tensor[0].float(), dim=(-2, -1)).mean()
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
