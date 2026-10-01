"""Steering strength in one unit for every method, on tune or held-out questions.

Unit: ``N`` is the natural recurrent-state norm over all GDN heads, at the
last prompt token of the tune prompts (chat template, no filler), averaged
over prompts. A method that injects ``D`` (the rank-k truncation, the full
matrix, or the clamp's ``sigma u w^T``) uses the factor ``scale * N / ||D||``,
so at scale 1 the injected tensor has the norm of a natural state. ``||D||``
is the Frobenius norm over all heads.

Add methods write into the zero state before the first prompt token
(``prompt_position=0``); per-head normalization is off. Clamp holds each
head's component along ``u`` on every prompt and generated token.

Scored per row: concept (Lingua for languages, the judge otherwise),
content quality from the judge (the ``answer_quality`` guide for languages),
and token 4-gram repetition. The chosen scale per method has the highest
concept rate among scales whose quality and repetition stay near the
unsteered baseline. Ties take the smaller scale.

    uv run python experiments/steering-scale/run.py --direction runs/directions/en-ru/direction \\
        --feature ru --questions runs/pairs-v2/eval_questions.jsonl --output runs/scale/en-ru
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path

import torch

from hybrid_steering import (
    LANGUAGE_FEATURES,
    Runner,
    chat_prompts,
    final_states,
    gdn_layers,
    load_direction,
    load_runtime,
    truncate_direction,
)
from hybrid_steering.judge import score_rows, score_steering
from hybrid_steering.report import summary_section, write_page
from hybrid_steering.runtime import batched, write_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forgetting"))
from questions import eval_split  # noqa: E402

METHODS = {"rank1": 1, "rank2": 2, "full": None, "clamp": 1}
SCALES = (0.25, 0.5, 1, 2, 4, 8, 16)
QUALITY_DROP = 0.5
REPETITION_RISE = 0.1


def natural_norm(model, tokenizer, questions: list[str]) -> float:
    runner = Runner(model, tokenizer, gdn_layers(model), normalize=False)
    states = final_states(runner, chat_prompts(tokenizer, questions))
    squares = sum(state.float().pow(2).sum((1, 2, 3)) for state in states.values())
    return float(squares.sqrt().mean())


def injected_norm(direction: dict, rank: int | None) -> float:
    truncated = truncate_direction(direction, rank)
    return math.sqrt(sum(float(tensor.pow(2).sum()) for tensor in truncated.values()))


def repetition(token_ids: list[int], n: int = 4) -> float:
    grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


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


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def score(rows: list[dict], feature: str, batch_size: int) -> None:
    language = feature in LANGUAGE_FEATURES
    score_rows(rows, feature, prompt_field="question", batch_size=batch_size)
    if language:
        judgments = score_steering(
            [(row["question"], row["response"]) for row in rows],
            "answer_quality",
            batch_size=batch_size,
        )
        for row, judgment in zip(rows, judgments, strict=True):
            row["content_quality"] = judgment.content_quality if judgment else None
            row["evaluable"] = judgment.evaluable if judgment else None
            row["flags"] = list(judgment.flags) if judgment else []
    for row in rows:
        if language:
            row["hit"] = int(row["concept_score"] >= 1)
        elif row.get("evaluable") is False:
            row["hit"] = 0
        elif row.get("concept_score") is None:
            row["hit"] = None
        else:
            row["hit"] = int(row["concept_score"] >= 2)


def summarize(rows: list[dict]) -> list[dict]:
    cells = {}
    for row in rows:
        cells.setdefault((row["method"], row["scale"]), []).append(row)
    summary = []
    for (method, scale), items in sorted(cells.items(), key=lambda item: (item[0][0], item[0][1])):
        hits = [row["hit"] for row in items if row["hit"] is not None]
        quality = [row["content_quality"] for row in items if row["content_quality"] is not None]
        low, high = wilson(sum(hits), len(hits))
        summary.append(
            {
                "method": method,
                "scale": scale,
                "n": len(hits),
                "concept_rate": sum(hits) / len(hits) if hits else math.nan,
                "ci_low": low,
                "ci_high": high,
                "quality": sum(quality) / len(quality) if quality else math.nan,
                "repetition": sum(row["repetition"] for row in items) / len(items),
                "unevaluable": sum(row.get("evaluable") is False for row in items) / len(items),
            }
        )
    return summary


def choose(summary: list[dict]) -> dict[str, dict]:
    baseline = next(item for item in summary if item["method"] == "baseline")
    chosen = {}
    for method in METHODS:
        candidates = [
            item
            for item in summary
            if item["method"] == method
            and item["quality"] >= baseline["quality"] - QUALITY_DROP
            and item["repetition"] <= baseline["repetition"] + REPETITION_RISE
        ]
        if candidates:
            chosen[method] = max(
                candidates, key=lambda item: (item["concept_rate"], -item["scale"])
            )
    return chosen


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
            prompt_position=None if clamp else 0,
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
