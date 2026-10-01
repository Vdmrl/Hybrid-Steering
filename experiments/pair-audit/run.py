"""Audit the concept pairs before they become steering directions.

For every concept on the Hub dataset: pair count, distinct prompts, token
lengths of both sides, and the share of texts that name the target concept.
A negative text that names the target ("no deity guides me") means the pair
contrasts affirming and denying the concept, not having and lacking it.
Random pairs are printed in full for reading.

    uv run python experiments/pair-audit/run.py --output runs/pair-audit
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
from html import escape
from pathlib import Path

from hybrid_steering.direction import load_concept_pairs, read_pairs
from hybrid_steering.report import write_page

CONCEPTS = {
    "atheistic_framing-theistic_framing": (
        "pairs.jsonl",
        r"\b(god|divine|faith|deity|supernatural|creator|prayer|spiritual)",
    ),
    "binary_framing-probabilistic_framing": (
        "pairs.jsonl",
        r"(probabil|likelihood|chance|uncertain|odds|spectrum)",
    ),
    "factual_reporting-fictional_narrative": (
        "pairs.jsonl",
        r"(fiction|narrative|story|stories|storytelling|plot|character)",
    ),
    "isolated_framing-comparative_framing": (
        "pairs.jsonl",
        r"(compar|contrast|unlike|parallel|relative to|versus)",
    ),
    "plain-theistic_framing": (
        "pairs.jsonl",
        r"\b(god|divine|faith|deity|supernatural|creator|prayer|spiritual)",
    ),
    "plain-probabilistic_framing": (
        "pairs.jsonl",
        r"(probabil|likelihood|chance|uncertain|odds|spectrum)",
    ),
    "plain-fictional_narrative": (
        "pairs.jsonl",
        r"(fiction|narrative|story|stories|storytelling|plot|character)",
    ),
    "plain-comparative_framing": (
        "pairs.jsonl",
        r"(compar|contrast|unlike|parallel|relative to|versus)",
    ),
    "technical": ("pairs_n218.jsonl", None),
    "en-ru": ("pairs.jsonl", None),
    "en-fr": ("pairs.jsonl", None),
    "en-zh": ("pairs.jsonl", None),
    "en-ar": ("pairs.jsonl", None),
}
NEGATION = r"\b(rather than|instead of|without|do not|don't|not|no|never)\b"


def share(texts: list[str], pattern: str | None) -> float | None:
    if pattern is None:
        return None
    return sum(bool(re.search(pattern, text, re.I)) for text in texts) / len(texts)


def audit(concept: str, rows: list[dict], tokenizer, pattern: str | None) -> dict:
    positive = [row["positive_text"] for row in rows]
    negative = [row["negative_text"] for row in rows]
    positive_tokens = [len(ids) for ids in tokenizer(positive).input_ids]
    negative_tokens = [len(ids) for ids in tokenizer(negative).input_ids]
    ratios = sorted(p / n for p, n in zip(positive_tokens, negative_tokens, strict=True))
    prompts = {row.get("source_question") or row.get("user_prompt") for row in rows}
    return {
        "concept": concept,
        "pairs": len(rows),
        "prompts": len(prompts - {None}),
        "positive_tokens": statistics.median(positive_tokens),
        "negative_tokens": statistics.median(negative_tokens),
        "ratio_p10": ratios[len(ratios) // 10],
        "ratio_median": statistics.median(ratios),
        "ratio_p90": ratios[9 * len(ratios) // 10],
        "target_words_positive": share(positive, pattern),
        "target_words_negative": share(negative, pattern),
        "negation_negative": share(negative, NEGATION),
    }


def number(value) -> str:
    if value is None:
        return "–"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def table(stats: list[dict]) -> str:
    columns = list(stats[0])
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(number(row[column]))}</td>" for column in columns) + "</tr>"
        for row in stats
    )
    return f"<h2>Pair statistics</h2><table><tr>{head}</tr>{body}</table>"


def samples(concept: str, rows: list[dict], count: int, seed: int) -> str:
    picked = random.Random(seed).sample(rows, min(count, len(rows)))
    items = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('source_question') or row.get('user_prompt') or ''))}</td>"
        f"<td style='text-align:left'>{escape(row['positive_text'])}</td>"
        f"<td style='text-align:left'>{escape(row['negative_text'])}</td>"
        "</tr>"
        for row in picked
    )
    return (
        f"<details><summary><b>{escape(concept)}</b></summary><table>"
        "<tr><th>prompt</th><th>positive (target)</th><th>negative (source)</th></tr>"
        f"{items}</table></details>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--pairs", type=Path, help="directory with local <concept>/pairs.jsonl")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stats, sections = [], []
    for concept, (name, pattern) in CONCEPTS.items():
        local = args.pairs / concept / name if args.pairs else None
        if local and local.exists():
            rows = read_pairs(local)
        elif concept.startswith("plain-"):
            continue
        else:
            rows = load_concept_pairs(concept, name)
        stats.append(audit(concept, rows, tokenizer, pattern))
        sections.append(samples(concept, rows, args.samples, args.seed))
    write_page(
        args.output / "report.html",
        "Concept pair audit",
        [table(stats), "<h2>Random pairs</h2>", *sections],
    )
    print(f"wrote {args.output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
