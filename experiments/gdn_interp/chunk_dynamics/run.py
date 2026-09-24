import argparse
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .capture import ChunkCapture
from .plots import write_plots

MODEL = "Qwen/Qwen3.5-9B"
DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"
DOCUMENTS = 200
MAX_TOKENS = 8193


def texts(tokenizer: PreTrainedTokenizerBase, dataset: str, subset: str, documents: int, length: int = MAX_TOKENS) -> list[torch.Tensor]:
    stream = load_dataset(dataset, subset, split="train", streaming=True).shuffle(seed=0, buffer_size=1_000)
    accepted = []
    for row in stream:
        ids = tokenizer(row["text"], truncation=True, max_length=length, return_tensors="pt").input_ids[0]
        if len(ids) == length:
            accepted.append(ids)
        if len(accepted) == documents:
            return accepted
    raise RuntimeError("dataset stream ended before enough documents reached max_tokens")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", type=Path, help="existing metrics.parquet; only regenerate plots")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--subset", default=SUBSET)
    parser.add_argument("--documents", type=int, default=DOCUMENTS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    if args.metrics:
        output = args.output or args.metrics.parent
        output.mkdir(parents=True, exist_ok=True)
        write_plots(pd.read_parquet(args.metrics), output)
        return
    if args.output is None:
        parser.error("--output is required unless --metrics is supplied")
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(args.model))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map="cuda").eval()
    probe = ChunkCapture(model)
    documents, rows = texts(tokenizer, args.dataset, args.subset, args.documents), []
    for start in range(0, len(documents), args.batch_size):
        batch = torch.stack(documents[start : start + args.batch_size]).to(probe.device)
        for layer, position, document, head, metric, value in probe.measure(batch):
            rows.append((start + document, layer, position, head, metric, value))
        print(f"{min(start + args.batch_size, len(documents))}/{len(documents)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=("document", "layer", "position", "head", "metric", "value"))
    frame.to_parquet(args.output / "metrics.parquet", index=False)
    write_plots(frame, args.output)


if __name__ == "__main__":
    main()
