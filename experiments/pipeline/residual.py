"""Additive residual-stream steering at one decoder layer.

The direction is the mean last-token residual of the target text minus the
source text, one vector per layer. Generation applies it on every token.
``add`` writes ``scale`` times the raw vector. ``clamp`` replaces the
projection on that direction with the same vector.
"""

from __future__ import annotations

import argparse
import json
import random
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
from hybrid_steering.mamba import _restore_sequence_axis
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


def token_mask(hidden: torch.Tensor, args, kwargs) -> torch.Tensor | None:
    """Return a 2D padding mask aligned with ``hidden``, ignoring causal masks."""
    candidates = []
    if kwargs:
        for key in ("mamba_attention_mask", "attention_mask"):
            mask = kwargs.get(key)
            if torch.is_tensor(mask):
                candidates.append(mask)
    for value in args:
        if torch.is_tensor(value):
            candidates.append(value)
    for candidate in candidates:
        if candidate.ndim == 2 and tuple(candidate.shape) == tuple(hidden.shape[:2]):
            return candidate
    return None


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
    positions = (encoded.attention_mask.long().cumsum(-1) - 1).clamp_min(0)
    with torch.inference_mode():
        output = model(
            **encoded, position_ids=positions, output_hidden_states=True, use_cache=False
        )
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


def residual_delta(
    hidden: torch.Tensor, vector: torch.Tensor, scale: float, mode: str
) -> torch.Tensor:
    """Per-token update along ``vector``.

    ``add`` is ``scale * v``. ``clamp`` removes the projection on the unit
    direction and writes ``scale * v``: ``x - (x·û) û + scale * v``.
    """
    if mode not in {"add", "clamp"}:
        raise ValueError("residual mode must be add or clamp")
    flat = vector.detach().to(device=hidden.device, dtype=torch.float32).reshape(-1)
    if mode == "add":
        return (flat * scale).to(dtype=hidden.dtype)
    unit = flat / flat.norm().clamp_min(1e-8)
    coeff = (hidden.float() * unit).sum(dim=-1, keepdim=True)
    return (scale * flat - coeff * unit).to(dtype=hidden.dtype)


def install_falcon_norm_hook(model) -> None:
    """Keep a one-token decode from collapsing to ``[batch, hidden]``.

    Falcon's gated norm squeezes that token. Adding the result to attention
    then broadcasts the batch axis into the sequence axis.
    """
    for layer in decoder_layers(model):
        mixer = getattr(layer, "mamba", None)
        if (
            mixer is None
            or not getattr(mixer, "mamba_rms_norm", False)
            or getattr(mixer, "_hybrid_norm_hook", False)
        ):
            continue
        mixer.norm.register_forward_hook(_restore_sequence_axis)
        mixer._hybrid_norm_hook = True


def falcon_greedy(model, tokenizer, texts: list[str], max_new_tokens: int) -> torch.Tensor:
    """Batched greedy decode with an explicit left-pad mask.

    ``model.generate`` builds a causal mask that Falcon's SDPA rejects when a
    batch mixes sequence lengths. The returned tensor is padded to
    ``max_new_tokens``.
    """
    install_falcon_norm_hook(model)
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(
        device
    )
    ids, mask = encoded.input_ids, encoded.attention_mask
    if (mask.sum(-1) < 2).any():
        raise ValueError("Falcon prompts need at least two tokens")
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    cache = None
    if ids.shape[1] > 1:
        cache = model(
            input_ids=ids[:, :-1],
            attention_mask=mask[:, :-1],
            position_ids=positions[:, :-1],
            use_cache=True,
            logits_to_keep=1,
        ).past_key_values
    out = model(
        input_ids=ids[:, -1:],
        attention_mask=mask,
        position_ids=positions[:, -1:],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    cache, logits = out.past_key_values, out.logits[:, -1].to(device)
    pad = tokenizer.pad_token_id
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = torch.as_tensor(eos if isinstance(eos, list) else [eos], device=device)
    generated = torch.full((len(texts), max_new_tokens), pad, device=device, dtype=torch.long)
    finished = torch.zeros(len(texts), dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        token = torch.where(finished, pad, logits.argmax(-1))
        generated[:, step] = token
        finished |= torch.isin(token, eos_ids)
        if finished.all() or step + 1 == max_new_tokens:
            break
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
        out = model(
            input_ids=token[:, None],
            attention_mask=mask,
            position_ids=(mask.long().sum(-1) - 1)[:, None],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache, logits = out.past_key_values, out.logits[:, -1].to(device)
    return generated


class ResidualRunner:
    """Steer one residual vector on every token of one decoder layer."""

    def __init__(
        self, model, tokenizer, vector: torch.Tensor, layer: int, mode: str = "add"
    ) -> None:
        layers = decoder_layers(model)
        if layer < 0 or layer >= len(layers):
            raise ValueError(f"layer {layer} is outside 0..{len(layers) - 1}")
        if mode not in {"add", "clamp"}:
            raise ValueError("residual mode must be add or clamp")
        flat = vector.detach().float().reshape(-1)
        if flat.numel() != hidden_size(model):
            raise ValueError("residual vector does not match the model hidden size")
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.vector = flat
        self.mode = mode
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
            delta = residual_delta(hidden, vector, scale, self.mode)
            mask = token_mask(hidden, args, kwargs)
            if mask is not None:
                delta = delta * mask.to(dtype=hidden.dtype).unsqueeze(-1)
            hidden.add_(delta)
            return output

        handle = None if scale == 0 else module.register_forward_hook(hook, with_kwargs=True)
        try:
            with torch.inference_mode():
                if getattr(self.model.config, "model_type", None) == "falcon_h1":
                    return falcon_greedy(self.model, self.tokenizer, texts, max_new_tokens).cpu()
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
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        write_residual(directory, args.model, target, source, rows, vectors)
        norms = {str(layer): float(vector.norm()) for layer, vector in sorted(vectors.items())}
        (args.output / "residual_norms.json").write_text(json.dumps(norms, indent=2) + "\n")
        print(f"wrote {directory}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


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


PROBE_QUESTIONS = 30
PROBE_SCALE = 0.25
PROBE_HEADER = (
    "Answer the question below. The text after it is unrelated background; ignore it.\n\n"
)


def middle_layers(count: int) -> list[int]:
    """A few decoder layers across the middle half, not the ends."""
    if count < 1:
        raise ValueError("model has no decoder layers")
    start, stop = count // 4, (3 * count) // 4
    step = max(1, (stop - start) // 4)
    layers = list(range(start, stop + 1, step))
    return layers or [count // 2]


def matched_delta(vector: torch.Tensor, scale: float, activation_norm: float) -> torch.Tensor:
    """``scale * ||h|| * v / ||v||``, the forgetting residual probe."""
    flat = vector.detach().float().reshape(-1)
    return scale * activation_norm * flat / flat.norm().clamp_min(1e-8)


def _probe_questions(path: Path) -> list[str]:
    """First 30 of the forgetting tune split: seed 20261002, tune size 50."""
    rows = read_jsonl(path)
    if len(rows) <= 50:
        raise SystemExit(f"{path} has {len(rows)} questions, need more than 50")
    tune = random.Random(20261002).sample(rows, len(rows))[:50]
    return [row["question"] for row in tune[:PROBE_QUESTIONS]]


def _probe_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": PROBE_HEADER + question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _activation_norms(model, tokenizer, texts: list[str], layers: list[int]) -> dict[int, float]:
    tokenizer.padding_side = "left"
    encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    encoded = encoded.to(next(model.parameters()).device)
    positions = (encoded.attention_mask.long().cumsum(-1) - 1).clamp_min(0)
    with torch.inference_mode():
        output = model(
            **encoded, position_ids=positions, output_hidden_states=True, use_cache=False
        )
    hidden = output.hidden_states
    if not hidden or len(hidden) < 2:
        raise RuntimeError("model did not return decoder hidden states")
    index = encoded.attention_mask.sum(-1) - 1
    rows = torch.arange(index.shape[0], device=index.device)
    norms = {}
    for layer in layers:
        states = hidden[layer + 1][rows, index].float()
        norms[layer] = float(states.norm(dim=-1).mean())
    return norms


def probe_main(args) -> None:
    """Pick a middle layer the way the forgetting residual probe does.

    Thirty tune questions, scale 0.25, the norm-matched vector on every token.
    The winner is the highest Lingua rate, then the lower repetition, then the
    layer closer to the middle of the network.
    """
    choice = args.output / "layer.json"
    if choice.exists():
        print(json.dumps(json.loads(choice.read_text(encoding="utf-8"))), flush=True)
        return
    direction, _manifest, _, _ = load_direction(args.direction)
    questions = _probe_questions(args.questions)
    model, tokenizer = load_runtime(args.model)
    layers = decoder_layers(model)
    probed = [layer for layer in middle_layers(len(layers)) if layer in direction]
    if not probed:
        raise SystemExit("direction has none of the middle layers")
    texts = [_probe_prompt(tokenizer, question) for question in questions]
    norms = _activation_norms(model, tokenizer, texts, probed)
    detector = concept_detector(args.feature)
    cells = []
    for layer in probed:
        delta = matched_delta(direction[layer], PROBE_SCALE, norms[layer])
        module = layers[layer]
        device = next(model.parameters()).device
        encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
        encoded = encoded.to(device)
        pads = (encoded.attention_mask == 0).sum(1).tolist()

        def hook(_module, _args, _kwargs, output, delta=delta.to(device), pads=pads):
            hidden = output[0] if isinstance(output, tuple) else output
            update = delta.to(dtype=hidden.dtype)
            if hidden.shape[1] == 1:
                hidden.add_(update)
                return output
            mask = hidden.new_ones(hidden.shape[0], hidden.shape[1])
            for row, pad in enumerate(pads):
                mask[row, :pad] = 0
            hidden.add_(update * mask.unsqueeze(-1))
            return output

        handle = module.register_forward_hook(hook, with_kwargs=True)
        try:
            with torch.inference_mode():
                if getattr(model.config, "model_type", None) == "falcon_h1":
                    fresh = falcon_greedy(model, tokenizer, texts, args.max_new_tokens)
                else:
                    sequences = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    fresh = sequences[:, encoded.input_ids.shape[1] :]
        finally:
            handle.remove()
        concepts, reps = [], []
        for token_row in fresh.detach().cpu():
            ids = [int(token) for token in token_row if int(token) != tokenizer.pad_token_id]
            concepts.append(int(detector.detects(tokenizer.decode(ids, skip_special_tokens=True))))
            reps.append(repetition(ids))
        concept, rep = sum(concepts) / len(concepts), sum(reps) / len(reps)
        cells.append(
            {
                "layer": layer,
                "scale": PROBE_SCALE,
                "where": "all",
                "activation_norm": norms[layer],
                "concept": concept,
                "repetition": rep,
            }
        )
        print(f"layer {layer}: concept={concept:.3f} rep={rep:.3f}", flush=True)
    centre = len(layers) / 2
    winner = max(
        cells,
        key=lambda cell: (cell["concept"], -cell["repetition"], -abs(cell["layer"] - centre)),
    )
    if winner["concept"] <= 0:
        raise SystemExit("concept stays at 0 on every probed layer")
    payload = {
        "layer": winner["layer"],
        "scale": PROBE_SCALE,
        "where": "all",
        "activation_norm": winner["activation_norm"],
        "concept": winner["concept"],
        "questions": len(questions),
        "cells": cells,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    choice.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"layer": winner["layer"], "concept": winner["concept"]}), flush=True)


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
    probe = sub.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--model", default="Qwen/Qwen3.5-9B")
    probe.add_argument("--direction", type=Path, required=True)
    probe.add_argument("--feature", required=True)
    probe.add_argument("--questions", type=Path, required=True)
    probe.add_argument("--max-new-tokens", type=int, default=128)
    probe.set_defaults(func=probe_main)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
