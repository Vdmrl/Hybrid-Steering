"""Concept persistence across filler tokens, for every steering method.

Prompt: one user turn ``HEADER + question + "\\n\\n" + filler[:L]``, then the
assistant header. Add methods write the scaled direction into the GDN state
once, right after the question's last token, so the question itself is read
unsteered and L filler tokens plus the assistant header follow before the
answer. Clamp holds the direction on every token; it is the reference that
does not forget. ``--lengths 0`` is the scale-selection run.

Units follow steering-scale: factor = ``scale * N / ||D_method||`` with ``N``
measured on the tune questions at L = 0.

Per row: concept hit, content quality, and repetition, as in steering-scale.
Per (method, scale, filler, L), on the first ``--mechanics-questions``
questions, per GDN layer and head at the last prompt token against the
unsteered prompt: the state difference norm, the unsteered state norm, the
cosine between head outputs, and the norm of the output difference.

    uv run python experiments/filler-decay/run.py --direction runs/directions/en-ru/direction \\
        --feature ru --questions runs/pairs-v2/eval_questions.jsonl --output runs/decay/en-ru \\
        --lengths 0 --scales 0.25 0.5 1 2 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from hybrid_steering import Runner, gdn_layers, load_direction, load_runtime
from hybrid_steering.runtime import import_path, token_prefixes, write_jsonl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "forgetting"))
from fillers import FILLERS  # noqa: E402
from questions import eval_split  # noqa: E402

scale_tools = import_path(HERE.parent / "steering-scale" / "run.py")

HEADER = "Answer the question below. The text after it is unrelated background; ignore it.\n\n"
LENGTHS = (0, 32, 64, 128, 256, 512, 1024, 2048)
METHODS = scale_tools.METHODS


def layout(tokenizer, question: str, filler: str) -> tuple[str, int]:
    """Chat prompt and the number of tokens up to and including the question's end."""
    content = HEADER + question + ("\n\n" + filler if filler else "")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    end = text.index(question) + len(question)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, sum(start < end for start, _ in offsets.offset_mapping)


def natural_norm(model, tokenizer, questions: list[str]) -> float:
    runner = Runner(model, tokenizer, gdn_layers(model), normalize=False)
    texts = [layout(tokenizer, question, "")[0] for question in questions]
    states = per_row(runner.forward(texts, prompt_position=None), len(texts))[1]
    return float(states.pow(2).sum((1, 2, 3, 4)).sqrt().mean())


def per_row(trace, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Last-token head outputs and final states, ``[row, layer, head, ...]``, layers sorted."""
    outputs = states = None
    for position, indices in trace.indices.items():
        layers = sorted(trace.states[position])
        out = torch.stack([trace.outputs[position][layer].float() for layer in layers], 1)
        state = torch.stack([trace.states[position][layer].float() for layer in layers], 1)
        if outputs is None:
            outputs = out.new_empty((count, *out.shape[1:]))
            states = state.new_empty((count, *state.shape[1:]))
        outputs[indices] = out
        states[indices] = state
    return outputs, states


def head_metrics(base, steered) -> dict[str, torch.Tensor]:
    """Means over rows, each ``[layer, head]``."""
    (base_out, base_state), (out, state) = base, steered
    return {
        "state_delta": (state - base_state).flatten(3).norm(dim=-1).mean(0).cpu(),
        "state_norm": base_state.flatten(3).norm(dim=-1).mean(0).cpu(),
        "output_cosine": torch.nn.functional.cosine_similarity(out, base_out, dim=-1).mean(0).cpu(),
        "output_delta": (out - base_out).norm(dim=-1).mean(0).cpu(),
        "output_norm": base_out.norm(dim=-1).mean(0).cpu(),
    }


def width(length: int, batch_size: int, token_budget: int) -> int:
    return max(1, min(batch_size, token_budget // (length + 96)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("tune", "held"), default="tune")
    parser.add_argument("--count", type=int, help="first questions of the split")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    scales = parser.add_mutually_exclusive_group(required=True)
    scales.add_argument("--scales", type=float, nargs="+", help="same grid for every method")
    scales.add_argument("--chosen", type=Path, help="summary.json with a chosen scale per method")
    parser.add_argument("--multipliers", type=float, nargs="+", default=[1.0])
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--fillers", type=int, nargs="+", default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--token-budget", type=int, default=262144)
    parser.add_argument("--mechanics-questions", type=int, default=32)
    parser.add_argument("--judge-batch-size", type=int, default=64)
    args = parser.parse_args()

    tune, held = eval_split(args.questions)
    questions = [row["question"] for row in (tune if args.split == "tune" else held)]
    questions = questions[: args.count] if args.count else questions
    if args.chosen:
        chosen = json.loads(args.chosen.read_text())["chosen"]
        grid = {
            method: sorted({chosen[method]["scale"] * m for m in args.multipliers})
            for method in args.methods
            if method in chosen
        }
    else:
        grid = {method: sorted(args.scales) for method in args.methods}

    direction, manifest, _, _ = load_direction(args.direction)
    model, tokenizer = load_runtime(args.model)
    norm = natural_norm(model, tokenizer, [row["question"] for row in tune])
    factors = {
        method: norm / scale_tools.injected_norm(direction, METHODS[method]) for method in grid
    }
    prompts = {}
    for filler in args.fillers:
        cuts = token_prefixes(tokenizer, FILLERS[filler], args.lengths)
        for length in args.lengths:
            prompts[filler, length] = [
                layout(tokenizer, question, cuts[length] if length else "")
                for question in questions
            ]

    plain = Runner(model, tokenizer, gdn_layers(model), normalize=False)
    runners = {
        method: Runner.from_direction(
            model,
            tokenizer,
            direction,
            rank=None if method == "clamp" else METHODS[method],
            normalize=False,
            intervention="clamp" if method == "clamp" else "add",
        )
        for method in grid
    }

    rows, heads = [], {}
    for (filler, length), items in prompts.items():
        size = width(length, args.batch_size, args.token_budget)
        jobs = [("baseline", 0.0, index) for index in range(len(questions))] + [
            (method, scale, index)
            for method, values in grid.items()
            for scale in values
            for index in range(len(questions))
        ]
        for method in ["baseline", *grid]:
            mine = [job for job in jobs if job[0] == method]
            runner = plain if method == "baseline" else runners[method]
            for batch in [mine[i : i + size] for i in range(0, len(mine), size)]:
                tokens = runner.generate(
                    [items[index][0] for _, _, index in batch],
                    scale=torch.tensor([scale * factors.get(method, 0.0) for _, scale, _ in batch]),
                    prompt_position=(
                        torch.tensor([items[index][1] for _, _, index in batch])
                        if method in grid and method != "clamp"
                        else None
                    ),
                    max_new_tokens=args.max_new_tokens,
                )
                for (_, scale, index), row in zip(batch, tokens, strict=True):
                    ids = [int(t) for t in row if int(t) != tokenizer.pad_token_id]
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
                            "repetition": scale_tools.repetition(ids),
                            "split": args.split,
                            "target": manifest.target,
                            "source": manifest.source,
                        }
                    )

        probe = items[: args.mechanics_questions]
        texts = [text for text, _ in probe]
        base = per_row(plain.forward(texts, prompt_position=None), len(texts))
        for method, values in grid.items():
            for scale in values:
                trace = runners[method].forward(
                    texts,
                    scale=scale * factors[method],
                    prompt_position=(
                        None if method == "clamp" else torch.tensor([p for _, p in probe])
                    ),
                )
                heads[method, scale, filler, length] = head_metrics(
                    base, per_row(trace, len(texts))
                )
                del trace
        del base
        torch.cuda.empty_cache()
        print(f"filler {filler}, L={length}: generated", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"layers": gdn_layers(model), "cells": heads, "factors": factors},
        args.output / "heads.pt",
    )
    write_jsonl(args.output / "rows.jsonl", rows)
    scale_tools.score(rows, args.feature, args.judge_batch_size)
    write_jsonl(args.output / "rows.jsonl", rows)
    meta = {
        "feature": args.feature,
        "split": args.split,
        "questions": len(questions),
        "natural_norm": norm,
        "factors": factors,
        "grid": grid,
        "lengths": args.lengths,
        "fillers": args.fillers,
        "header": HEADER,
    }
    cells = []
    for filler in args.fillers:
        for length in args.lengths:
            subset = [r for r in rows if r["filler"] == filler and r["length"] == length]
            for cell in scale_tools.summarize(subset):
                cells.append({"filler": filler, "length": length, **cell})
    summary = {**meta, "cells": cells}
    if args.lengths == [0]:
        summary["chosen"] = scale_tools.choose(
            [{k: v for k, v in cell.items() if k not in ("filler", "length")} for cell in cells]
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
