"""GDN state rank and token-boundary transition metrics.

Smoke measures positions 1, 2 and 4 on one short random sequence. A full run
passes a Hugging Face model and the long position grid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from hybrid_steering.capture import CAPTURE_POSITIONS, ChunkCapture, rank_metrics
from hybrid_steering.runtime import load_runtime


def rank_rows(
    model, input_ids: torch.Tensor, layers: list[int], positions: tuple[int, ...]
) -> list[dict[str, float | int | str]]:
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    rows: list[dict[str, float | int | str]] = []
    start = 0
    for end in positions:
        with torch.inference_mode():
            model(
                input_ids=input_ids[:, start:end],
                attention_mask=torch.ones(
                    len(input_ids), end, device=input_ids.device, dtype=torch.long
                ),
                past_key_values=cache,
                use_cache=True,
            )
        states = torch.stack([cache.layers[layer].recurrent_states[0] for layer in layers])
        # states: [layer, batch, head, key, value] -> [batch, layer, head, key, value]
        measured = rank_metrics(states.permute(1, 0, 2, 3, 4))
        for metric, values in measured.items():
            for document, layer_index, head in torch.cartesian_prod(
                torch.arange(values.shape[0]),
                torch.arange(values.shape[1]),
                torch.arange(values.shape[2]),
            ).tolist():
                rows.append(
                    {
                        "document": document,
                        "layer": layers[layer_index],
                        "position": end,
                        "head": head,
                        "metric": metric,
                        "value": float(values[document, layer_index, head].cpu()),
                    }
                )
        start = end
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--documents", type=int, default=1)
    parser.add_argument("--positions", type=int, nargs="+")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    positions = tuple(args.positions or (CAPTURE_POSITIONS if not args.smoke else (1, 2, 4)))
    if args.smoke:
        args.model = "tiny"
        args.documents = 1
    model, tokenizer = load_runtime(args.model)
    length = max(positions) + 1
    text = "token " * length
    encoded = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=length, return_tensors="pt"
    )
    input_ids = encoded.input_ids.to(next(model.parameters()).device)
    if input_ids.shape[1] < length:
        pad = torch.arange(1, length - input_ids.shape[1] + 1, device=input_ids.device).view(1, -1)
        input_ids = torch.cat((input_ids, pad), dim=1)
    input_ids = input_ids.expand(args.documents, -1).contiguous()
    layers = [
        index for index, layer in enumerate(model.model.layers) if hasattr(layer, "linear_attn")
    ]
    capture = ChunkCapture(model)
    transitions = [
        {
            "layer": layer,
            "position": position,
            "document": document,
            "head": head,
            "metric": metric,
            "value": value,
        }
        for layer, position, document, head, metric, value in capture.measure(input_ids, positions)
    ]
    ranks = rank_rows(model, input_ids, layers, positions)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"positions": list(positions), "transitions": transitions, "ranks": ranks}
    (args.output / "metrics.json").write_text(json.dumps(payload) + "\n")
    if not transitions or not ranks:
        raise SystemExit("chunk dynamics produced no rows")
    if any(math.isnan(row["value"]) for row in transitions + ranks):
        raise SystemExit("chunk dynamics produced NaN")
    print(f"transitions={len(transitions)} ranks={len(ranks)}", flush=True)


if __name__ == "__main__":
    main()
