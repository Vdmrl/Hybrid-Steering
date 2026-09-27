"""Prepare paired coherent, shuffled, and repeated long-text controls."""

from __future__ import annotations

import argparse
import random
import re
from bisect import bisect_left
from itertools import pairwise
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MIN_PARAGRAPH_TOKENS = 32
MAX_PARAGRAPH_TOKENS = 512
MIN_SENTENCE_TOKENS = 16
SOURCE_CHARS_PER_TOKEN = 10
MIN_SOURCE_CHARS = 20_000


def sentence_chunks(text: str, tokenizer) -> tuple[list[int], list[list[int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoded["input_ids"]
    starts = [start for start, _ in encoded["offset_mapping"]]
    boundaries = [match.end() for match in re.finditer(r"[.!?][\"'’”)]*\s+", text)]
    indices = sorted(
        {0, len(token_ids), *(bisect_left(starts, boundary) for boundary in boundaries)}
    )
    return token_ids, [
        token_ids[start:end] for start, end in pairwise(indices) if end > start
    ]


def repeated_control(
    text: str, tokenizer, length: int, rng: random.Random
) -> list[int]:
    candidates = []
    for paragraph in re.split(r"\n\s*\n", text):
        ids = tokenizer(paragraph.strip(), add_special_tokens=False)["input_ids"]
        if MIN_PARAGRAPH_TOKENS <= len(ids) <= min(MAX_PARAGRAPH_TOKENS, length):
            candidates.append(ids)
    if not candidates:
        candidates = [
            ids
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if len(ids := tokenizer(sentence, add_special_tokens=False)["input_ids"])
            >= MIN_SENTENCE_TOKENS
        ]
    if not candidates:
        raise ValueError("fragment has no usable paragraph")
    paragraph = rng.choice(candidates)
    return (paragraph * ((length + len(paragraph) - 1) // len(paragraph)))[:length]


def controls(
    text: str, tokenizer, length: int, seed: int
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    token_ids, sentences = sentence_chunks(text, tokenizer)
    if len(token_ids) < length or len(sentences) < 2:
        raise ValueError("fragment is too short")
    coherent = token_ids[:length]
    used = []
    count = 0
    for sentence in sentences:
        if count >= length:
            break
        used.append(sentence[: length - count])
        count += len(used[-1])
    if len(used) < 2:
        raise ValueError("fragment has fewer than two sentences in the token window")
    rng.shuffle(used)
    shuffled = [token for sentence in used for token in sentence][:length]
    repeated = repeated_control(text, tokenizer, length, rng)
    if not (len(coherent) == len(shuffled) == len(repeated) == length):
        raise AssertionError("control lengths differ")
    return coherent, shuffled, repeated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare paired long-text controls for GDN state analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset", default="claran/pg19-sample")
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--title-column", default="short_book_title")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--length", type=int, default=10_000)
    parser.add_argument(
        "--coherent-only",
        action="store_true",
        help="skip shuffled/repeated controls when only source text is needed",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="timestamped artifacts/state_dynamics run directory",
    )
    args = parser.parse_args()
    output = args.run_dir / "inputs" / "sequences.pt"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(
        args.dataset, args.config or None, split=args.split, streaming=True
    )
    rng = random.Random(args.seed)
    records = []
    for source_index, row in enumerate(dataset):
        text = row["text"]
        span = min(
            len(text),
            max(args.length * SOURCE_CHARS_PER_TOKEN, MIN_SOURCE_CHARS),
        )
        if len(text) < span:
            continue
        low = min(len(text) // 10, max(0, len(text) - span))
        start = rng.randint(low, len(text) - span)
        fragment = text[start : start + span]
        if args.coherent_only:
            ids = tokenizer(fragment, add_special_tokens=False)["input_ids"]
            if len(ids) < args.length:
                continue
            variants = (ids[: args.length],)
            data_types = ("coherent",)
        else:
            try:
                variants = controls(
                    fragment, tokenizer, args.length, args.seed + source_index
                )
            except ValueError:
                continue
            data_types = ("coherent", "shuffled", "repeated")
        source_id = f"source-{source_index:05d}"
        title = row.get(args.title_column, source_id)
        for data_type, ids in zip(data_types, variants, strict=True):
            records.append(
                {
                    "text_id": source_id,
                    "data_type": data_type,
                    "title": title,
                    "input_ids": torch.tensor(ids, dtype=torch.int32),
                }
            )
        print(f"{len(records) // len(data_types)}/{args.count}: {title}")
        if len(records) == len(data_types) * args.count:
            break
    expected_records = args.count if args.coherent_only else 3 * args.count
    if len(records) != expected_records:
        raise RuntimeError(f"only prepared {len(records)} usable documents")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "seed": args.seed,
            "length": args.length,
            "coherent_only": args.coherent_only,
            "records": records,
        },
        output,
    )
    print(f"wrote {len(records)} sequences to {output}")


if __name__ == "__main__":
    main()
