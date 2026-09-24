"""Experimental one-time additive and coordinate-clamp GDN steering."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from gdn_interp import LanguageDetector, truncate_svd
from generate import MODEL, prompts


def directions(artifact: dict[str, Any], layers: set[int] | None, heads: set[int] | None, rank: int) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    result, targets = {}, {}
    for layer, records in artifact["layers"].items():
        if layers is not None and layer not in layers:
            continue
        delta = torch.stack([head["mean_delta"] for head in records]).float()
        delta = truncate_svd(delta, rank)
        direction = delta / torch.linalg.matrix_norm(delta, dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(delta.dtype).tiny)
        if heads is not None:
            direction[[head for head in range(len(direction)) if head not in heads]] = 0
        result[layer] = direction
        # D is EN-RU, hence Russian has the lower natural coordinate.
        targets[layer] = torch.stack([head["mean_b"] for head in records]).float()
    if not result:
        raise ValueError("no layers selected")
    if not any(torch.count_nonzero(d) for d in result.values()):
        raise ValueError("all steering directions are zero; check head selection")
    return result, targets


def intervene(cache: Any, direction: dict[int, torch.Tensor], targets: dict[int, torch.Tensor], mode: str, fraction: float) -> dict[str, float]:
    stats = {}
    for layer, d in direction.items():
        state = cache.layers[layer].recurrent_states[0]
        assert state is not None
        d = d.to(state)
        before = (state.float() * d).sum((-2, -1))
        stats[f"layer_{layer}_state_fro"] = float(torch.linalg.matrix_norm(state.float(), dim=(-2, -1)).mean())
        if mode == "add":
            state.add_((-fraction * d).to(state.dtype))
        else:
            ru = (targets[layer].to(state) * d).sum((-2, -1))
            target = before + fraction * (ru - before)
            state.add_(((target - before)[..., None, None] * d).to(state.dtype))
        after = (state.float() * d).sum((-2, -1))
        stats[f"layer_{layer}_target_coordinate"] = float(ru.mean()) if mode == "clamp" else float("nan")
        stats[f"layer_{layer}_coordinate_before"] = float(before.mean())
        stats[f"layer_{layer}_coordinate_after"] = float(after.mean())
    return stats


def decode(model: Any, output: Any, mask: torch.Tensor, n: int, eos: int | None, pad: int) -> torch.Tensor:
    token, cache, generated = output.logits[:, -1].argmax(-1), output.past_key_values, []
    finished = torch.zeros_like(token, dtype=torch.bool)
    for _ in range(n):
        generated.append(token)
        finished |= token.eq(eos) if eos is not None else False
        if finished.all(): break
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), 1)
        output = model(input_ids=torch.where(finished, torch.full_like(token, pad), token)[:, None], attention_mask=mask, past_key_values=cache, use_cache=True)
        token, cache = output.logits[:, -1].argmax(-1), output.past_key_values
    return torch.stack(generated, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("add", "clamp"), default="clamp")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=0, help="0 is full rank")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--heads", type=int, nargs="+")
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    target = artifact.get("concept_b", artifact.get("language_b", "ru"))
    if not isinstance(target, str):
        raise ValueError("artifact concept_b must be a string")
    detector = LanguageDetector(target)
    direction, targets = directions(artifact, set(args.layers) if args.layers else None, set(args.heads) if args.heads else None, args.rank)
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(args.model)); tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    device = next(model.parameters()).device
    rows = list(load_dataset("rajpurkar/squad", split="validation").shuffle(seed=42).select(range(args.questions)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(prompts(tokenizer, [row["question"] for row in batch]), padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                prefill = model(**encoded, use_cache=True)
                stats = intervene(prefill.past_key_values, direction, targets, args.mode, args.fraction)
                generated = decode(model, prefill, encoded.attention_mask, args.max_new_tokens, tokenizer.eos_token_id, tokenizer.pad_token_id)
            for row, tokens in zip(batch, generated, strict=True):
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                out.write(json.dumps({"question": row["question"], "response": text, "detected_language": detector.label(text), "mode": args.mode, "fraction": args.fraction, "rank": args.rank, "state": stats}) + "\n")


if __name__ == "__main__": main()
