"""Steering strength in one unit for every method, on tune or held-out questions.

Unit: ``N`` is the natural recurrent-state norm over all GDN heads, at the
last prompt token of the tune prompts (chat template, no filler), averaged
over prompts. A method that injects ``D`` (the rank-k truncation, the full
matrix, or the clamp's ``sigma u w^T``) uses the factor ``scale * N / ||D||``,
so at scale 1 the injected tensor has the norm of a natural state. ``||D||``
is the Frobenius norm over all heads.

Add methods write into the state once at ``--prompt-position`` (default 0, the
zero state before the first prompt token); per-head normalization is off. Clamp holds each
head's component along ``u`` on every prompt and generated token.

Scored per row: concept (Lingua for languages, the judge otherwise),
content quality from the judge (the ``answer_quality`` guide for languages),
and token 4-gram repetition. The chosen scale per method has the highest
concept rate among scales whose quality and repetition stay near the
unsteered baseline. Ties take the smaller scale.

    uv run python experiments/steering-scale/run.py --direction runs/directions/en-ru/direction \\
        --feature ru --questions runs/pairs/eval_questions.jsonl --output runs/steering-scale/en-ru
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

import torch

from hybrid_steering import Runner, chat_prompts, gdn_layers, load_direction, load_runtime
from hybrid_steering.report import summary_section, write_page
from hybrid_steering.runtime import batched, write_jsonl
from hybrid_steering.scoring import (
    METHODS,
    QUALITY_DROP,
    REPETITION_RISE,
    choose,
    injected_norm,
    natural_norm,
    repetition,
    score,
    summarize,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forgetting"))
from questions import eval_split  # noqa: E402

SCALES = (0.25, 0.5, 1, 2, 4, 8, 16)


def generate(runner, tokenizer, texts, scales, *, prompt_position, max_new_tokens, batch_size):
    """Rows of (text index, scale, response, repetition) for every text and scale."""
    jobs = [(index, scale) for scale in scales for index in range(len(texts))]
    results = []
    for batch in batched(jobs, batch_size):
        tokens = runner.generate(
            [texts[index] for index, _ in batch],
            scale=torch.tensor([scale for _, scale in batch]),
            prompt_position=prompt_position,
            max_new_tokens=max_new_tokens,
        )
        for (index, scale), row in zip(batch, tokens, strict=True):
            ids = [int(t) for t in row if int(t) != tokenizer.pad_token_id]
            results.append(
                (index, scale, tokenizer.decode(ids, skip_special_tokens=True), repetition(ids))
            )
    return results


def report(path: Path, rows: list[dict], summary: list[dict], chosen: dict, meta: dict) -> None:
    def table(items: list[dict]) -> str:
        columns = list(items[0])
        head = "".join(f"<th>{escape(column)}</th>" for column in columns)
        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{item[c]:.3f}</td>" if isinstance(item[c], float) else f"<td>{item[c]}</td>"
                for c in columns
            )
            + "</tr>"
            for item in items
        )
        return f"<table><tr>{head}</tr>{body}</table>"

    steered = [row for row in rows if row["method"] != "baseline"]
    examples = []
    for method, item in chosen.items():
        picked = [r for r in steered if r["method"] == method and r["scale"] == item["scale"]][:4]
        largest = max(r["scale"] for r in steered if r["method"] == method)
        picked += [r for r in steered if r["method"] == method and r["scale"] == largest][:2]
        for row in picked:
            examples.append(
                f"<tr><td>{method}</td><td>{row['scale']}</td><td>{row['concept_score']}</td>"
                f"<td>{row['content_quality']}</td><td style='text-align:left'>"
                f"{escape(row['question'])}</td><td style='text-align:left'>"
                f"{escape(row['response'])}</td></tr>"
            )
    sections = [
        f"<p>{escape(json.dumps(meta))}</p>",
        "<h2>Chosen scale per method</h2>" + table(list(chosen.values())),
        summary_section(steered, "scale", "hit", "method", "Concept rate"),
        summary_section(steered, "scale", "content_quality", "method", "Content quality 0–4"),
        summary_section(steered, "scale", "repetition", "method", "Token 4-gram repetition"),
        "<h2>All cells</h2>" + table(summary),
        "<h2>Examples: chosen scale, then the largest</h2><table><tr><th>method</th>"
        "<th>scale</th><th>concept</th><th>quality</th><th>question</th><th>response</th>"
        f"</tr>{''.join(examples)}</table>",
    ]
    write_page(path, f"Steering scale: {meta['feature']}", sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("tune", "held"), default="tune")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument("--scales", type=float, nargs="+", default=SCALES)
    parser.add_argument("--prompt-position", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--judge-batch-size", type=int, default=64)
    args = parser.parse_args()

    tune, held = eval_split(args.questions)
    questions = [row["question"] for row in (tune if args.split == "tune" else held)]
    direction, manifest, _, _ = load_direction(args.direction)
    model, tokenizer = load_runtime(args.model)
    texts = chat_prompts(tokenizer, questions)
    norm = natural_norm(model, tokenizer, [row["question"] for row in tune])
    factors = {method: norm / injected_norm(direction, rank) for method, rank in METHODS.items()}

    plain = Runner(model, tokenizer, gdn_layers(model), normalize=False)
    raw = [
        ("baseline", item)
        for item in generate(
            plain,
            tokenizer,
            texts,
            [0.0],
            prompt_position=None,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    ]
    for method in args.methods:
        clamp = method == "clamp"
        runner = Runner.from_direction(
            model,
            tokenizer,
            direction,
            rank=None if clamp else METHODS[method],
            normalize=False,
            intervention="clamp" if clamp else "add",
        )
        applied = [scale * factors[method] for scale in args.scales]
        for index, value, response, rep in generate(
            runner,
            tokenizer,
            texts,
            applied,
            prompt_position=None if clamp else args.prompt_position,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        ):
            raw.append((method, (index, args.scales[applied.index(value)], response, rep)))
        print(f"{method}: generated", flush=True)

    rows = [
        {
            "method": method,
            "scale": scale,
            "factor": factors.get(method, 0.0),
            "question": questions[index],
            "response": response,
            "repetition": rep,
            "split": args.split,
            "target": manifest.target,
            "source": manifest.source,
        }
        for method, (index, scale, response, rep) in raw
    ]
    score(rows, args.feature, args.judge_batch_size)
    write_jsonl(args.output / "rows.jsonl", rows)
    summary = summarize(rows)
    chosen = choose(summary)
    meta = {
        "feature": args.feature,
        "split": args.split,
        "prompt_position": args.prompt_position,
        "questions": len(questions),
        "natural_norm": norm,
        "factors": factors,
        "quality_drop": QUALITY_DROP,
        "repetition_rise": REPETITION_RISE,
    }
    (args.output / "summary.json").write_text(
        json.dumps({**meta, "cells": summary, "chosen": chosen}, indent=1) + "\n"
    )
    report(args.output / "report.html", rows, summary, chosen, meta)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
