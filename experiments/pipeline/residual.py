"""Additive residual-stream steering at one decoder layer.

The direction is the mean last-token residual of the target text minus the
source text, one vector per layer. Generation adds ``scale`` times that vector
on every token. There is no norm matching and no clamp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_steering import load_direction, load_runtime
from hybrid_steering.artifacts import save_direction
from hybrid_steering.detect import concept_detector, is_language
from hybrid_steering.direction import (
    concept_sides,
    load_concept_pairs,
    read_pairs,
    target_and_source,
)
from hybrid_steering.models import DirectionManifest
from hybrid_steering.runtime import batched, chat_prompts, read_jsonl
from hybrid_steering.scoring import repetition


def hidden_size(model) -> int:
    config = model.config
    for item in (getattr(config, "text_config", None), config):
        size = getattr(item, "hidden_size", None)
        if size:
            return int(size)
    raise ValueError("model config has no hidden size")


def decoder_layers(model):
    root = model.model
    if hasattr(root, "layers"):
        return root.layers
    language = getattr(root, "language_model", None)
    if language is not None and hasattr(language, "layers"):
        return language.layers
    raise ValueError("model has no decoder layers")


def last_layer_states(model, tokenizer, texts: list[str]) -> dict[int, torch.Tensor]:
    """Last real token of each decoder layer, shape ``[batch, hidden]``."""
    encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    encoded = encoded.to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = output.hidden_states
    if not hidden or len(hidden) < 2:
        raise RuntimeError("model did not return decoder hidden states")
    index = encoded.attention_mask.sum(-1) - 1
    rows = torch.arange(index.shape[0], device=index.device)
    return {layer: states[rows, index].float().cpu() for layer, states in enumerate(hidden[1:])}


def collect_residual(
    model, tokenizer, pairs: list[tuple[str, str]], batch_size: int
) -> dict[int, torch.Tensor]:
    """Mean target-minus-source residual at the last token of every layer."""
    if batch_size < 1 or not pairs:
        raise ValueError("pairs and a positive batch_size are required")
    total: dict[int, torch.Tensor] | None = None
    seen = 0
    for batch in batched(pairs, batch_size):
        target = last_layer_states(model, tokenizer, [text for text, _ in batch])
        source = last_layer_states(model, tokenizer, [text for _, text in batch])
        if set(target) != set(source):
            raise RuntimeError("target and source layers differ")
        if total is None:
            total = {layer: torch.zeros_like(vector[0]) for layer, vector in target.items()}
        for layer, vector in target.items():
            total[layer] += (vector - source[layer]).sum(0)
        seen += len(batch)
    assert total is not None
    return {layer: vector / seen for layer, vector in total.items()}


def write_residual(
    directory: Path,
    model_id: str,
    target: str,
    source: str,
    rows: list[dict],
    vectors: dict[int, torch.Tensor],
) -> None:
    save_direction(
        directory,
        vectors,
        DirectionManifest(
            model_id=model_id,
            target=target,
            source=source,
            example_ids=[
                str(row.get("pair_id") or f"pair-{index}") for index, row in enumerate(rows)
            ],
            decoder_layer_indices=sorted(vectors),
            state_shapes={layer: list(vector.shape) for layer, vector in vectors.items()},
        ),
    )


class ResidualRunner:
    """Add one residual vector on every token of one decoder layer."""

    def __init__(self, model, tokenizer, vector: torch.Tensor, layer: int) -> None:
        layers = decoder_layers(model)
        if layer < 0 or layer >= len(layers):
            raise ValueError(f"layer {layer} is outside 0..{len(layers) - 1}")
        flat = vector.detach().float().reshape(-1)
        if flat.numel() != hidden_size(model):
            raise ValueError("residual vector does not match the model hidden size")
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.vector = flat
        tokenizer.padding_side = "left"

    def generate(
        self, texts: list[str], scale: float = 1.0, max_new_tokens: int = 64, **_unused
    ) -> torch.Tensor:
        if not texts or max_new_tokens < 1:
            raise ValueError("texts and max_new_tokens are required")
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            texts, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(device)
        module = decoder_layers(self.model)[self.layer]
        vector = self.vector.to(device=device)

        def hook(_module, args, kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            mask = kwargs.get("attention_mask")
            if mask is None:
                mask = next(
                    (
                        value
                        for value in args
                        if torch.is_tensor(value)
                        and value.ndim == 2
                        and value.shape[:2] == hidden.shape[:2]
                    ),
                    None,
                )
            delta = vector.to(dtype=hidden.dtype)
            if mask is not None and tuple(mask.shape[:2]) == tuple(hidden.shape[:2]):
                delta = delta * mask.to(dtype=hidden.dtype).unsqueeze(-1)
            hidden.add_(delta * scale)
            return output

        handle = None if scale == 0 else module.register_forward_hook(hook, with_kwargs=True)
        try:
            with torch.inference_mode():
                sequences = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if handle is not None:
                handle.remove()
        fresh = sequences[:, encoded.input_ids.shape[1] :]
        padded = torch.full(
            (len(texts), max_new_tokens), self.tokenizer.pad_token_id, dtype=torch.long
        )
        take = min(max_new_tokens, fresh.shape[1])
        padded[:, :take] = fresh[:, :take].cpu()
        return padded


def _load_rows(args) -> tuple[str, str, list[dict]]:
    source, target = concept_sides(args.concept, args.source, args.target)
    rows = read_pairs(args.jsonl) if args.jsonl else load_concept_pairs(args.concept)
    rows = rows[: args.pairs or None]
    if not rows:
        raise SystemExit("no pairs")
    return source, target, rows


def extract_main(args) -> None:
    source, target, rows = _load_rows(args)
    model, tokenizer = load_runtime(args.model)
    vectors = collect_residual(
        model, tokenizer, [target_and_source(row) for row in rows], args.batch_size
    )
    directory = args.output / "direction"
    write_residual(directory, args.model, target, source, rows, vectors)
    norms = {str(layer): float(vector.norm()) for layer, vector in sorted(vectors.items())}
    (args.output / "residual_norms.json").write_text(json.dumps(norms, indent=2) + "\n")
    print(f"wrote {directory}", flush=True)


def _concept_score(rows: list[dict], feature: str, settings: Path | None) -> None:
    if is_language(feature):
        detector = concept_detector(feature)
        for row in rows:
            row["concept_score"] = int(detector.detects(row["response"]))
        return
    from hybrid_steering.judge import score_rows

    score_rows(rows, feature, settings_path=settings)


def _mean(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def pick_main(args) -> None:
    direction, _manifest, _, _ = load_direction(args.direction)
    prompts = [row["prompt"] for row in read_jsonl(args.prompts)][: args.prompts_limit or None]
    if len(prompts) < 4:
        raise SystemExit("need at least 4 prompts to pick a layer")
    model, tokenizer = load_runtime(args.model)
    texts = chat_prompts(tokenizer, prompts)

    def run(layer: int, scale: float) -> list[dict]:
        runner = ResidualRunner(model, tokenizer, direction[layer], layer)
        tokens = runner.generate(texts, scale=scale, max_new_tokens=args.max_new_tokens)
        rows = []
        for prompt, token_row in zip(prompts, tokens, strict=True):
            ids = [int(token) for token in token_row if int(token) != tokenizer.pad_token_id]
            rows.append(
                {
                    "prompt": prompt,
                    "response": tokenizer.decode(ids, skip_special_tokens=True),
                    "repetition": repetition(ids),
                }
            )
        _concept_score(rows, args.feature, args.judge_config)
        return rows

    cells = []
    baseline_rows = run(sorted(direction)[0], 0.0)
    baseline_concept = _mean(baseline_rows, "concept_score")
    baseline_repetition = _mean(baseline_rows, "repetition")
    cells.append(
        {
            "layer": None,
            "scale": 0.0,
            "concept": baseline_concept,
            "repetition": baseline_repetition,
        }
    )
    print(f"baseline concept={baseline_concept:.3f}", flush=True)
    for layer in sorted(direction):
        for scale in args.scales:
            rows = run(layer, scale)
            concept, rep = _mean(rows, "concept_score"), _mean(rows, "repetition")
            cells.append({"layer": layer, "scale": scale, "concept": concept, "repetition": rep})
            print(f"layer {layer} scale {scale:g}: concept={concept:.3f} rep={rep:.3f}", flush=True)
    steered = [cell for cell in cells if cell["layer"] is not None]
    stable = [cell for cell in steered if cell["repetition"] <= baseline_repetition + 0.1]
    winner = max(
        stable or steered,
        key=lambda cell: (cell["concept"], -cell["scale"], -cell["layer"]),
    )
    chosen = {
        "layer": winner["layer"],
        "scale": winner["scale"],
        "concept": winner["concept"],
        "beats_baseline": winner["concept"] > baseline_concept and bool(stable),
    }
    payload = {"baseline_concept": baseline_concept, "chosen": chosen, "cells": cells}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "layer_choice.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(chosen), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--model", default="Qwen/Qwen3.5-9B")
    extract.add_argument("--concept", required=True)
    extract.add_argument("--jsonl", type=Path)
    extract.add_argument("--pairs", type=int, default=0)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--source")
    extract.add_argument("--target")
    extract.set_defaults(func=extract_main)
    pick = sub.add_parser("pick-layer")
    pick.add_argument("--output", type=Path, required=True)
    pick.add_argument("--model", default="Qwen/Qwen3.5-9B")
    pick.add_argument("--direction", type=Path, required=True)
    pick.add_argument("--feature", required=True)
    pick.add_argument("--prompts", type=Path, required=True)
    pick.add_argument("--prompts-limit", type=int, default=0)
    pick.add_argument("--scales", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    pick.add_argument("--max-new-tokens", type=int, default=96)
    pick.add_argument("--judge-config", type=Path)
    pick.set_defaults(func=pick_main)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
