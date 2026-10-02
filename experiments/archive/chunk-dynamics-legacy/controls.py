import argparse
from pathlib import Path
from typing import cast

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .capture import CHUNK_SIZE, ChunkCapture
from .run import DATASET, DOCUMENTS, MODEL, SUBSET, texts

Row = tuple[int, int, int, int, str, float]
CONTROL_POSITIONS = (*range(1, 32), 64, 128, 256)
CONTROL_LENGTH = max(CONTROL_POSITIONS) + 1


def top_u(state: torch.Tensor) -> torch.Tensor:
    return torch.svd_lowrank(state.float().transpose(-2, -1), q=1, niter=2)[0][..., :, 0]


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left, right, dim=-1, eps=torch.finfo(right.dtype).tiny).abs()


def append(rows: list[Row], layer: int, position: int, values: dict[str, torch.Tensor]) -> None:
    for metric, tensor in values.items():
        for document, head in torch.cartesian_prod(torch.arange(tensor.shape[0]), torch.arange(tensor.shape[1])).tolist():
            rows.append((layer, position, document, head, metric, float(tensor[document, head].cpu())))


def selected_events(probe: ChunkCapture, input_ids: torch.Tensor):
    cache, pending, start = DynamicCache(config=probe.model.config), {}, 0
    while start < CONTROL_LENGTH:
        targets = [position for position in CONTROL_POSITIONS if start < position <= start + CHUNK_SIZE]
        end = targets[0] if targets else min(start + CHUNK_SIZE, CONTROL_LENGTH)
        if start < 32:
            end = start + 1
        captured = probe.capture(input_ids[:, start:end], end, cache, recurrent=start > 0 and end - start == 1)
        for layer, tensors in captured.items():
            if layer in pending:
                yield layer, pending.pop(layer), tensors
            if end in CONTROL_POSITIONS:
                pending[layer] = end, tensors[-1]
        start = end


def first_pass(probe: ChunkCapture, input_ids: torch.Tensor, sums: dict[int, tuple[torch.Tensor, int, torch.Tensor, int]]) -> list[Row]:
    cache, pending, rows, start = DynamicCache(config=probe.model.config), {}, [], 0
    while start < CONTROL_LENGTH:
        targets = [position for position in CONTROL_POSITIONS if start < position <= start + CHUNK_SIZE]
        end = targets[0] if targets else min(start + CHUNK_SIZE, CONTROL_LENGTH)
        if start < 32:
            end = start + 1
        captured = probe.capture(input_ids[:, start:end], end, cache, recurrent=start > 0 and end - start == 1)
        for layer, (key, value, beta, gate, state) in captured.items():
            del key, beta, gate
            value = value.float()
            value_sum = value.sum((0, 1))
            old_value_sum, value_count, covariance, u_count = sums.get(layer, (torch.zeros_like(value_sum), 0, torch.zeros(value.shape[2], value.shape[3], value.shape[3], device=value.device), 0))
            if layer in pending:
                position, old_state = pending.pop(layer)
                u, current = top_u(old_state), value[:, 0]
                documents = torch.randperm(len(value), device=value.device)
                tokens = torch.randint(value.shape[1], (len(value),), device=value.device)
                random_value = value[documents, tokens]
                covariance += torch.einsum("bhv,bhw->hvw", u, u)
                u_count += len(u)
                append(rows, layer, position, {"sim_v": cosine(u, current), "sim_v_random": cosine(u, random_value), "sim_v_to_random_v": cosine(current, random_value)})
            sums[layer] = old_value_sum + value_sum, value_count + value.shape[0] * value.shape[1], covariance, u_count
            if end in CONTROL_POSITIONS:
                pending[layer] = end, state
        start = end
    return rows


def global_controls(probe: ChunkCapture, input_ids: torch.Tensor, means: dict[int, torch.Tensor], axes: dict[int, torch.Tensor]) -> list[Row]:
    rows: list[Row] = []
    for layer, (position, state), (_, value, _, _, _) in selected_events(probe, input_ids):
        u, value = top_u(state), value[:, 0].float()
        append(rows, layer, position, {
            "sim_v": cosine(u, value),
            "sim_v_mean": cosine(u, means[layer]),
            "sim_v_centered": cosine(u, value - means[layer]),
            "sim_v_global_axis": cosine(axes[layer].unsqueeze(0), value),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("first", "global"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--subset", default=SUBSET)
    parser.add_argument("--documents", type=int, default=DOCUMENTS)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(args.model))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    probe, documents = ChunkCapture(model), texts(tokenizer, args.dataset, args.subset, args.documents, CONTROL_LENGTH)
    if args.mode == "first":
        if args.statistics is None:
            parser.error("--statistics is required for first")
        sums: dict[int, tuple[torch.Tensor, int, torch.Tensor, int]] = {}
        rows: list[Row] = []
        for start in range(0, len(documents), args.batch_size):
            measured = first_pass(probe, torch.stack(documents[start : start + args.batch_size]).to(probe.device), sums)
            rows.extend((layer, position, start + document, head, metric, value) for layer, position, document, head, metric, value in measured)
            print(f"{min(start + args.batch_size, len(documents))}/{len(documents)}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=("layer", "position", "document", "head", "metric", "value")).to_parquet(args.output, index=False)
        torch.save({layer: (value.cpu(), value_count, covariance.cpu(), u_count) for layer, (value, value_count, covariance, u_count) in sums.items()}, args.statistics)
        return
    if args.mode == "global":
        if args.statistics is None:
            parser.error("--statistics is required for global")
        stored = torch.load(args.statistics, weights_only=True)
        means = {int(layer): value.to(probe.device) / value_count for layer, (value, value_count, _, _) in stored.items()}
        axes = {int(layer): torch.linalg.eigh((covariance / u_count).to(probe.device))[1][..., -1] for layer, (_, _, covariance, u_count) in stored.items()}
    rows: list[Row] = []
    for start in range(0, len(documents), args.batch_size):
        batch = torch.stack(documents[start : start + args.batch_size]).to(probe.device)
        measured = global_controls(probe, batch, means, axes)
        rows.extend((layer, position, start + document, head, metric, value) for layer, position, document, head, metric, value in measured)
        print(f"{min(start + args.batch_size, len(documents))}/{len(documents)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=("layer", "position", "document", "head", "metric", "value")).to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
