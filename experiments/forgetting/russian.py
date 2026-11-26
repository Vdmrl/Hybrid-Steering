"""Russian persistence for Qwen and Falcon, plus cosine to the injected write.

The prompt matches the published sweep. An additive method writes the scaled
direction once, on the last question token, and the filler plus the assistant
header are the first text that sees it. ``state_delta`` is the difference of
recurrent states at the last prompt token. ``write_cosine`` compares that
difference with the matrix added at the write. It is not the difference at
filler length 0, because that state has already passed through the header.
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

LENGTHS = (0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
METHODS = {"rank1": 1, "rank2": 2, "full": None}
BEHAVIOR = (
    ("baseline", "#8E939A", "baseline"),
    ("rank1", "#1B7F72", "rank 1"),
    ("rank2", "#C4A35A", "rank 2"),
    ("full", "#D4723A", "full"),
)


def prompt_columns(prompt_position: int | torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Padded index of the first token that should see the steered state."""
    lengths = mask.long().sum(-1)
    padding = mask.shape[1] - lengths
    requested = torch.as_tensor(prompt_position, device=mask.device, dtype=torch.long)
    if requested.ndim == 0:
        requested = requested.expand(lengths.shape[0])
    if requested.shape != lengths.shape:
        raise ValueError("prompt_position must be a scalar or one value per prompt")
    columns = padding + torch.where(requested < 0, requested + lengths, requested)
    if ((columns < padding) | (columns > padding + lengths)).any():
        raise ValueError("prompt_position must resolve within every prompt")
    return columns


def write_matrix(
    direction: dict[int, torch.Tensor], layers: list[int], scale: float
) -> torch.Tensor:
    """Injected write, stacked in layer order. ``scale`` already includes the dose factor."""
    if sorted(direction) != layers:
        raise ValueError(f"direction layers {sorted(direction)} do not match state layers {layers}")
    return torch.stack([direction[layer].detach().float() * scale for layer in layers])


def write_cosine(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Cosine of each row with the injected write. ``delta`` has shape ``[row, ...]``."""
    flat = delta.detach().float().flatten(1)
    ref = reference.detach().float().reshape(-1)
    if flat.shape[1] != ref.numel():
        raise ValueError("delta and the write must contain the same number of elements")
    if float(ref.norm()) == 0.0:
        raise ValueError("the injected write is zero")
    return torch.nn.functional.cosine_similarity(flat, ref.expand_as(flat), dim=-1)


def _add(sums: dict[str, Any], key: str, value: torch.Tensor) -> None:
    value = value.detach().float().cpu()
    current = sums.get(key)
    sums[key] = value if current is None else current + value


def add_states(
    sums: dict[str, Any],
    base: torch.Tensor,
    steered: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    delta = steered.detach().float().cpu() - base.detach().float().cpu()
    _add(sums, "state_delta", delta.flatten(3).norm(dim=-1).sum(0))
    _add(sums, "state_norm", base.detach().float().cpu().flatten(3).norm(dim=-1).sum(0))
    _add(sums, "write_cosine", write_cosine(delta, reference).sum())
    sums["count"] = int(sums["count"]) + int(delta.shape[0])


def add_outputs(sums: dict[str, Any], base: torch.Tensor, steered: torch.Tensor) -> None:
    base = base.detach().float().cpu()
    steered = steered.detach().float().cpu()
    _add(sums, "output_cosine", torch.nn.functional.cosine_similarity(steered, base, dim=-1).sum(0))
    _add(sums, "output_delta", (steered - base).norm(dim=-1).sum(0))
    _add(sums, "output_norm", base.norm(dim=-1).sum(0))


def finish_states(sums: dict[str, Any]) -> dict[str, torch.Tensor]:
    count = int(sums["count"])
    if count < 1:
        raise ValueError("no states to summarize")
    return {key: value / count for key, value in sums.items() if isinstance(value, torch.Tensor)}


def state_cell(
    base: torch.Tensor, steered: torch.Tensor, reference: torch.Tensor
) -> dict[str, torch.Tensor]:
    sums: dict[str, Any] = {"count": 0}
    add_states(sums, base, steered, reference)
    return finish_states(sums)


def _span(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    cache: Any,
    start: int,
    end: int,
) -> tuple[Any, torch.Tensor | None]:
    if end <= start:
        return cache, None
    kwargs: dict[str, Any] = {
        "input_ids": ids[:, start:end],
        "attention_mask": mask[:, :end],
        "position_ids": positions[:, start:end],
        "use_cache": True,
        "logits_to_keep": 1,
    }
    if cache is not None:
        kwargs["past_key_values"] = cache
    out = model(**kwargs)
    return out.past_key_values, out.logits[:, -1].to(ids.device)


def prefill(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    prompt_position: int | torch.Tensor | None,
    scale: float | torch.Tensor,
    direction: dict[int, torch.Tensor],
) -> tuple[Any, torch.Tensor]:
    """Run a Falcon prompt. ``prompt_position`` is how many real tokens are already read."""
    from hybrid_steering.cache import apply_direction

    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    if prompt_position is None:
        cache, logits = _span(model, ids, mask, positions, None, 0, ids.shape[1])
    else:
        columns = prompt_columns(prompt_position, mask)
        cache, logits = None, None
        start = 0
        for end in sorted({int(value) for value in columns.tolist()}):
            cache, logits = _span(model, ids, mask, positions, cache, start, end)
            if cache is None:
                raise RuntimeError("the steering write resolved before any token")
            active = columns.eq(end)
            apply_direction(
                cache,
                direction,
                torch.as_tensor(scale, dtype=torch.float32, device=ids.device) * active,
                normalize=False,
            )
            start = end
        cache, logits = _span(model, ids, mask, positions, cache, start, ids.shape[1])
    if logits is None:
        raise RuntimeError("prompt produced no logits")
    return cache, logits


def _tokenize(
    tokenizer: Any, texts: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
    return batch.input_ids, batch.attention_mask


def falcon_generate(
    model: Any,
    tokenizer: Any,
    direction: dict[int, torch.Tensor],
    texts: list[str],
    prompt_position: torch.Tensor | None,
    scale: float | torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    ids, mask = _tokenize(tokenizer, texts, device)
    if (mask.sum(-1) < 2).any():
        raise ValueError("Falcon prompts need at least two tokens")
    cache, logits = prefill(model, ids, mask, prompt_position, scale, direction)
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


def stack_states(by_layer: dict[int, torch.Tensor]) -> tuple[list[int], torch.Tensor]:
    layers = sorted(by_layer)
    return layers, torch.stack([by_layer[layer].detach().float().cpu() for layer in layers], 1)


def falcon_states(
    model: Any,
    tokenizer: Any,
    direction: dict[int, torch.Tensor],
    texts: list[str],
    prompt_position: torch.Tensor | None,
    scale: float,
) -> tuple[list[int], torch.Tensor]:
    from hybrid_steering.cache import extract_recurrent

    device = next(model.parameters()).device
    ids, mask = _tokenize(tokenizer, texts, device)
    cache, _logits = prefill(model, ids, mask, prompt_position, scale, direction)
    layers, states = stack_states(extract_recurrent(cache))
    del cache
    return layers, states


def _rows_of(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _pairs(rows: list[dict], field: str, method: str) -> dict[int, list[tuple[int, float]]]:
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        if method != "baseline" and abs(float(row["scale"]) - 1.0) > 1e-6:
            continue
        if row.get(field) is None:
            continue
        grouped[int(row["length"])].append((int(row["index"]), float(row[field])))
    return grouped


def _bootstrap(pairs: list[tuple[int, float]], seed: int) -> tuple[float, float, float]:
    by_question: dict[int, list[float]] = defaultdict(list)
    for index, value in pairs:
        by_question[index].append(value)
    keys = list(by_question)

    def mean(sample: list[int]) -> float:
        values = [item for key in sample for item in by_question[key]]
        return sum(values) / len(values)

    point = mean(keys)
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(keys, k=len(keys))) for _ in range(1000))
    return point, draws[25], draws[974]


def _behavior(rows: list[dict], field: str, method: str) -> list[dict]:
    points = []
    for length, pairs in sorted(_pairs(rows, field, method).items()):
        point, low, high = _bootstrap(pairs, zlib.crc32(f"{method}:{length}".encode()))
        points.append({"length": length, "score": point, "low": low, "high": high})
    return points


def _filler_mean(cells: dict, method: str, lengths: list[int]) -> tuple[list[dict], list[dict]]:
    norms: dict[int, list[float]] = defaultdict(list)
    cosines: dict[int, list[float]] = defaultdict(list)
    for (cell_method, scale, filler, length), cell in cells.items():
        if cell_method != method or int(length) not in lengths:
            continue
        origin = None
        for (other, other_scale, other_filler, other_length), other_cell in cells.items():
            if (
                other == method
                and other_scale == scale
                and other_filler == filler
                and int(other_length) == 0
            ):
                origin = float(other_cell["state_delta"].norm())
        if origin is None or origin == 0.0:
            raise ValueError(f"{method} has no state difference at filler length 0")
        norms[int(length)].append(float(cell["state_delta"].norm()) / origin)
        cosines[int(length)].append(float(cell["write_cosine"]))
    ratio = [
        {"length": length, "score": sum(norms[length]) / len(norms[length])}
        for length in lengths
        if norms[length]
    ]
    cosine = [
        {"length": length, "score": sum(cosines[length]) / len(cosines[length])}
        for length in lengths
        if cosines[length]
    ]
    return ratio, cosine


def plot_figure(output: Path) -> None:
    import sys

    import matplotlib.pyplot as plt

    folder = str(Path(__file__).resolve().parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from paper import band, style

    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    rows = _rows_of(output / "rows.jsonl")
    cells = torch.load(output / "heads.pt", map_location="cpu", weights_only=False)["cells"]
    falcon = meta.get("model_type") == "falcon_h1"
    field, ylabel, y_max = (
        ("language_pct", "percent", 100.0) if falcon else ("hit", "concept rate", 1.0)
    )
    lengths = [int(length) for length in meta["lengths"]]
    style()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    shown: list[dict] = []
    for method, color, label in BEHAVIOR:
        points = _behavior(rows, field, method)
        if not points:
            raise ValueError(f"{method} has no {field} scores")
        shown.extend(points)
        band(ax, points, color, label=label)
    right = ax.twinx()
    ratio, _full_cosine = _filler_mean(cells, "full", lengths)
    right.plot(
        [row["length"] + 1 for row in ratio],
        [row["score"] for row in ratio],
        color="#6E6E6E",
        linestyle=(0, (3.5, 2.2)),
        linewidth=1.6,
        label="state delta",
        zorder=1,
    )
    for method, color, label in BEHAVIOR[1:]:
        _ratio, cosine = _filler_mean(cells, method, lengths)
        right.plot(
            [row["length"] + 1 for row in cosine],
            [row["score"] for row in cosine],
            color=color,
            linestyle=(0, (1.2, 1.4)),
            linewidth=1.4,
            label=f"cosine, {label}",
            zorder=2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("filler tokens", labelpad=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03 * y_max, y_max * 1.04)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.5)
    ax.set_xlim(0.7, (max(lengths) + 1) * 1.4)
    ax.set_xticks([length + 1 for length in lengths])
    ax.set_xticklabels(
        [str(length) for length in lengths], rotation=30, ha="right", rotation_mode="anchor"
    )
    ax.minorticks_off()
    name = "Falcon" if falcon else "Qwen"
    ax.set_title(f"{name}, Russian, scale 1", loc="left", pad=8)
    right.set_ylim(-0.05, 1.05)
    right.set_ylabel(r"$\|\Delta S(L)\|/\|\Delta S(0)\|$ and cosine", color="#6E6E6E")
    right.tick_params(colors="#6E6E6E", length=3.5)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color("#6E6E6E")
    right.spines["top"].set_visible(False)
    right.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = right.get_legend_handles_labels()
    fig.legend(
        [*handles, *extra_handles],
        [*labels, *extra_labels],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=2.6,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    path = output / "figure.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    payload = {
        "behavior": {method: _behavior(rows, field, method) for method, _color, _label in BEHAVIOR},
        "state_ratio": {method: _filler_mean(cells, method, lengths)[0] for method in METHODS},
        "write_cosine": {method: _filler_mean(cells, method, lengths)[1] for method in METHODS},
    }
    (output / "curves.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(path, flush=True)


def _language_percent(detector: Any, text: str, target: str) -> float:
    # The published Falcon curve reads this confidence. The public detector only returns a label.
    if not text.strip():
        return 0.0
    for item in detector._detector.compute_language_confidence_values(text):
        if item.language.iso_code_639_1.name.lower() == target:
            return 100.0 * float(item.value)
    return 0.0


def score(output: Path, feature: str) -> None:
    from hybrid_steering.detect import concept_detector
    from hybrid_steering.runtime import write_jsonl

    rows = _rows_of(output / "rows.jsonl")
    detector = concept_detector(feature)
    target = feature.lower()
    for row in rows:
        text = str(row["response"]).strip()
        row["hit"] = int(detector.detects(text))
        row["language_pct"] = _language_percent(detector, text, target)
    write_jsonl(output / "rows.jsonl", rows)
    print(f"scored {output}", flush=True)


def _complete(
    rows: list[dict],
    cells: dict,
    filler: int,
    length: int,
    expected: int,
    methods: list[str],
    scales: list[float],
) -> bool:
    count = sum(1 for row in rows if int(row["filler"]) == filler and int(row["length"]) == length)
    if count != expected:
        return False
    return all(
        (method, float(scale), filler, length) in cells for method in methods for scale in scales
    )


def _save(output: Path, rows: list[dict], cells: dict, meta: dict, layers: list[int]) -> None:
    from hybrid_steering.runtime import write_jsonl

    output.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": layers, "cells": cells, "factors": meta["factors"]}, output / "heads.pt")
    write_jsonl(output / "rows.jsonl", rows)
    (output / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


@torch.inference_mode()
def generate(args: argparse.Namespace) -> None:
    from fillers import FILLERS
    from questions import eval_split
    from run import layout, natural_norm, per_row, width

    from hybrid_steering import gdn_layers, load_direction, load_runtime
    from hybrid_steering.mamba import MambaRunner
    from hybrid_steering.runner import Runner
    from hybrid_steering.runtime import token_prefixes
    from hybrid_steering.scoring import injected_norm, repetition
    from hybrid_steering.state import truncate_direction

    tune, held = eval_split(args.questions)
    questions = [row["question"] for row in (tune if args.split == "tune" else held)]
    questions = questions[: args.count] if args.count else questions
    if args.mechanics_questions < 1 or args.mechanics_questions > len(questions):
        raise ValueError("mechanics-questions must cover at least one question and fit the split")
    direction, manifest, _target, _source = load_direction(args.direction)
    model, tokenizer = load_runtime(args.model)
    falcon = getattr(model.config, "model_type", None) == "falcon_h1"
    if falcon:
        hook = MambaRunner(model, tokenizer)
        texts = [layout(tokenizer, question, "")[0] for question in tune]
        states = hook.final_states(texts)
        total = None
        for state in states.values():
            square = state.float().pow(2).flatten(1).sum(1)
            total = square if total is None else total + square
        norm = float(total.sqrt().mean())
    else:
        norm = natural_norm(model, tokenizer, [row["question"] for row in tune])
    factors = {method: norm / injected_norm(direction, METHODS[method]) for method in args.methods}
    prompts: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for filler in args.fillers:
        cuts = token_prefixes(tokenizer, FILLERS[filler], args.lengths)
        for length in args.lengths:
            prompts[filler, length] = [
                layout(tokenizer, question, cuts[length] if length else "")
                for question in questions
            ]
    if falcon:
        runners = {
            method: MambaRunner(model, tokenizer, direction, rank=METHODS[method], normalize=False)
            for method in args.methods
        }
        plain_direction: dict[int, torch.Tensor] = {}
    else:
        runners = {
            method: Runner.from_direction(
                model,
                tokenizer,
                direction,
                rank=METHODS[method],
                normalize=False,
                intervention="add",
            )
            for method in args.methods
        }
        plain = Runner(model, tokenizer, gdn_layers(model), normalize=False)
        plain_direction = {}
    output = args.output
    row_path = output / "rows.jsonl"
    head_path = output / "heads.pt"
    rows = _rows_of(row_path)
    saved = (
        torch.load(head_path, map_location="cpu", weights_only=False) if head_path.exists() else {}
    )
    cells = saved.get("cells", {})
    expected = len(questions) * (1 + len(args.methods) * len(args.scales))
    done = {
        (filler, length)
        for filler, length in prompts
        if _complete(rows, cells, filler, length, expected, args.methods, args.scales)
    }
    rows = [row for row in rows if (int(row["filler"]), int(row["length"])) in done]
    cells = {key: value for key, value in cells.items() if (int(key[2]), int(key[3])) in done}
    meta = {
        "feature": args.feature,
        "model": args.model,
        "model_type": getattr(model.config, "model_type", None),
        "split": args.split,
        "questions": len(questions),
        "mechanics_questions": args.mechanics_questions,
        "natural_norm": norm,
        "factors": factors,
        "lengths": list(args.lengths),
        "fillers": list(args.fillers),
        "methods": list(args.methods),
        "scales": list(args.scales),
        "target": manifest.target,
        "source": manifest.source,
        "direction": str(args.direction),
    }
    layers: list[int] = list(saved.get("layers", []))
    for (filler, length), items in prompts.items():
        if (filler, length) in done:
            print(f"skip filler {filler}, L={length}", flush=True)
            continue
        size = width(length, args.batch_size, args.token_budget)
        jobs = [("baseline", 0.0, index) for index in range(len(questions))] + [
            (method, scale, index)
            for method in args.methods
            for scale in args.scales
            for index in range(len(questions))
        ]
        for method in ["baseline", *args.methods]:
            mine = [job for job in jobs if job[0] == method]
            for batch in [mine[i : i + size] for i in range(0, len(mine), size)]:
                texts = [items[index][0] for _method, _scale, index in batch]
                ends = torch.tensor([items[index][1] for _method, _scale, index in batch])
                applied = torch.tensor(
                    [scale * factors.get(method, 0.0) for _method, scale, _index in batch]
                )
                if falcon:
                    tokens = falcon_generate(
                        model,
                        tokenizer,
                        plain_direction if method == "baseline" else runners[method].direction,
                        texts,
                        None if method == "baseline" else ends,
                        0.0 if method == "baseline" else applied,
                        args.max_new_tokens,
                    )
                else:
                    runner = plain if method == "baseline" else runners[method]
                    tokens = runner.generate(
                        texts,
                        scale=applied,
                        prompt_position=None if method == "baseline" else ends,
                        max_new_tokens=args.max_new_tokens,
                    )
                for (_method, scale, index), generated in zip(batch, tokens, strict=True):
                    ids = [
                        int(token) for token in generated if int(token) != tokenizer.pad_token_id
                    ]
                    rows.append(
                        {
                            "method": method,
                            "scale": scale,
                            "factor": factors.get(method, 0.0),
                            "filler": filler,
                            "length": length,
                            "index": index,
                            "question": questions[index],
                            "response": tokenizer.decode(ids, skip_special_tokens=True),
                            "repetition": repetition(ids),
                            "split": args.split,
                            "target": manifest.target,
                            "source": manifest.source,
                        }
                    )
        probe = items[: args.mechanics_questions]
        probe_texts = [text for text, _end in probe]
        probe_ends = torch.tensor([end for _text, end in probe])
        for method in args.methods:
            for scale in args.scales:
                sums: dict[str, Any] = {"count": 0}
                reference = None
                applied_scale = float(scale) * factors[method]
                truncated = (
                    runners[method].direction
                    if falcon
                    else truncate_direction(direction, METHODS[method])
                )
                for start in range(0, len(probe_texts), size):
                    chunk_texts = probe_texts[start : start + size]
                    chunk_ends = probe_ends[start : start + size]
                    if falcon:
                        base_layers, base_state = falcon_states(
                            model, tokenizer, {}, chunk_texts, None, 0.0
                        )
                        state_layers, state = falcon_states(
                            model, tokenizer, truncated, chunk_texts, chunk_ends, applied_scale
                        )
                        outputs = None
                        base_outputs = None
                    else:
                        base_trace = plain.forward(chunk_texts, prompt_position=None)
                        base_outputs, base_state = per_row(base_trace, len(chunk_texts))
                        trace = runners[method].forward(
                            chunk_texts, scale=applied_scale, prompt_position=chunk_ends
                        )
                        outputs, state = per_row(trace, len(chunk_texts))
                        base_layers = list(plain.layers)
                        state_layers = list(runners[method].layers)
                        del base_trace, trace
                    if base_layers != state_layers:
                        raise RuntimeError("baseline and steered states cover different layers")
                    if reference is None:
                        reference = write_matrix(truncated, state_layers, applied_scale)
                        layers = state_layers
                    add_states(sums, base_state, state, reference)
                    if outputs is not None and base_outputs is not None:
                        add_outputs(sums, base_outputs, outputs)
                    del base_state, state
                cells[method, float(scale), filler, length] = finish_states(sums)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _save(output, rows, cells, meta, layers)
        print(f"filler {filler}, L={length}: generated", flush=True)
    print(f"generated {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("runs/forgetting/questions.jsonl"))
    parser.add_argument("--feature", default="ru")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--split", choices=("tune", "held"), default="held")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(METHODS), default=["rank1", "rank2", "full"]
    )
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--fillers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--mechanics-questions", type=int, default=50)
    parser.add_argument("--stage", choices=("all", "generate", "score", "plot"), default="all")
    args = parser.parse_args()
    if args.stage in ("all", "generate"):
        generate(args)
    if args.stage in ("all", "score"):
        score(args.output, args.feature)
    if args.stage in ("all", "plot"):
        plot_figure(args.output)


if __name__ == "__main__":
    main()
