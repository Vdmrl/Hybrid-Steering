import argparse
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel
from transformers.cache_utils import LinearAttentionCacheLayerMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .capture import CAPTURE_POSITIONS
from .metrics import rank_metrics
from .run import DATASET, DOCUMENTS, MAX_TOKENS, MODEL, SUBSET, texts


def gdn_layers(model: PreTrainedModel) -> list[int]:
    layers = cast(torch.nn.ModuleList, model.get_submodule("model.layers"))
    return [index for index, layer in enumerate(layers) if hasattr(layer, "linear_attn")]


def measure(
    model: PreTrainedModel, input_ids: torch.Tensor, layers: list[int], positions: tuple[int, ...]
) -> list[tuple[int, int, int, int, str, float]]:
    cache, rows, start = DynamicCache(config=model.config), [], 0
    for end in positions:
        with torch.inference_mode():
            model(
                input_ids=input_ids[:, start:end],
                attention_mask=torch.ones(len(input_ids), end, device=input_ids.device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
        states = torch.stack(
            [cast(torch.Tensor, cast(LinearAttentionCacheLayerMixin, cache.layers[layer]).recurrent_states[0]) for layer in layers], dim=1
        )
        for metric, values in rank_metrics(states).items():
            for document, layer, head in torch.cartesian_prod(*[torch.arange(size) for size in values.shape]).tolist():
                rows.append((layers[layer], end, document, head, metric, float(values[document, layer, head].cpu())))
        start = end
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--subset", default=SUBSET)
    parser.add_argument("--documents", type=int, default=DOCUMENTS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-position", type=int, default=max(CAPTURE_POSITIONS))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    positions = tuple(position for position in CAPTURE_POSITIONS if position <= args.max_position)
    if not positions:
        parser.error("--max-position must be at least 1")
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(args.model))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map="cuda").eval()
    documents = texts(tokenizer, args.dataset, args.subset, args.documents, max(positions) + 1)
    layers, rows = gdn_layers(model), []
    for start in range(0, len(documents), args.batch_size):
        batch = torch.stack(documents[start : start + args.batch_size]).to(next(model.parameters()).device)
        for layer, position, document, head, metric, value in measure(model, batch, layers, positions):
            rows.append((start + document, layer, position, head, metric, value))
        print(f"{min(start + args.batch_size, len(documents))}/{len(documents)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=("document", "layer", "position", "head", "metric", "value")).to_parquet(args.output / "metrics.parquet", index=False)


if __name__ == "__main__":
    main()
