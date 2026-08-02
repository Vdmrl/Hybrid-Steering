"""Prepare clean donor pairs for Russian, optimism, and atomic sentences."""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

WIKISPLIT_ROWS = "https://datasets-server.huggingface.co/rows"
OPUS_TEST = (
    "https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/main/"
    "en-ru/test-00000-of-00001.parquet?download=true"
)
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")
POSITIVE = re.compile(
    r"\b(hope|opportunit|progress|improv|recover|success|promis|bright|"
    r"confident|resilien|potential|optimis)\w*",
    re.IGNORECASE,
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def ratio(left: str, right: str) -> float:
    return len(left.split()) / max(len(right.split()), 1)


def prepare_optimism(source: Path, count: int) -> list[dict[str, Any]]:
    candidates = []
    for row in jsonl(source):
        positive = str(row["positive_text"]).strip()
        negative = str(row["negative_text"]).strip()
        length_ratio = ratio(positive, negative)
        if (
            positive != negative
            and 0.7 <= length_ratio <= 1.3
            and 60 <= len(positive.split()) <= 450
            and not CYRILLIC.search(positive + negative)
        ):
            candidates.append(
                {
                    "source_id": row["id"],
                    "positive_text": positive,
                    "negative_text": negative,
                    "length_ratio": round(length_ratio, 3),
                }
            )
    return candidates[:count]


def wikisplit_page(offset: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": "cl-nagoya/wikisplit-pp",
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": 100,
        }
    )
    with urllib.request.urlopen(f"{WIKISPLIT_ROWS}?{query}", timeout=60) as response:
        return json.load(response)["rows"]


def prepare_atomic(count: int) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    for offset in range(0, 5000, 100):
        for wrapped in wikisplit_page(offset):
            row = wrapped["row"]
            complex_text = str(row["complex"]).strip()
            simple = row["simple_tokenized"]
            if not isinstance(simple, list) or len(simple) != 2:
                continue
            simple_text = " ".join(str(value).strip() for value in simple)
            length_ratio = ratio(simple_text, complex_text)
            key = complex_text.casefold()
            if (
                key in seen
                or float(row["entailment_prob"]) < 0.98
                or not 12 <= len(complex_text.split()) <= 60
                or not all(4 <= len(str(value).split()) <= 40 for value in simple)
                or not 0.75 <= length_ratio <= 1.3
                or "\n" in complex_text + simple_text
                or "?" in complex_text + simple_text
            ):
                continue
            seen.add(key)
            candidates.append(
                {
                    "source_id": f"wikisplit++:{row['id']}",
                    "positive_text": simple_text,
                    "negative_text": complex_text,
                    "length_ratio": round(length_ratio, 3),
                    "entailment_prob": round(float(row["entailment_prob"]), 6),
                }
            )
            if len(candidates) == count:
                return candidates
    raise RuntimeError(f"only {len(candidates)}/{count} clean WikiSplit++ pairs")


def prepare_russian(out: Path, count: int) -> list[dict[str, Any]]:
    parquet = out / "sources" / "opus100-en-ru-test.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    if not parquet.exists():
        urllib.request.urlretrieve(OPUS_TEST, parquet)
    frame = pd.read_parquet(parquet)
    candidates = []
    seen = set()
    for index, value in enumerate(frame["translation"]):
        english = str(value["en"]).strip()
        russian = str(value["ru"]).strip()
        key = english.casefold()
        length_ratio = len(russian) / max(len(english), 1)
        cyrillic_ratio = len(CYRILLIC.findall(russian)) / max(len(russian), 1)
        latin_ratio = len(LATIN.findall(english)) / max(len(english), 1)
        if (
            key in seen
            or not 6 <= len(english.split()) <= 80
            or not 0.6 <= length_ratio <= 2.2
            or cyrillic_ratio < 0.45
            or latin_ratio < 0.45
        ):
            continue
        seen.add(key)
        candidates.append(
            {
                "source_id": f"opus100:test:{index}",
                "positive_text": russian,
                "negative_text": english,
                "length_ratio": round(ratio(russian, english), 3),
            }
        )
        if len(candidates) == count:
            return candidates
    raise RuntimeError(f"only {len(candidates)}/{count} clean OPUS-100 pairs")


def sentence_length(text: str) -> float:
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    return len(text.split()) / max(len(sentences), 1)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row["positive_text"] for row in rows]
    negative = [row["negative_text"] for row in rows]
    return {
        "n": len(rows),
        "mean_length_ratio": round(
            sum(ratio(pos, neg) for pos, neg in zip(positive, negative, strict=True))
            / len(rows),
            3,
        ),
        "positive_words_per_sentence": round(
            sum(map(sentence_length, positive)) / len(rows), 3
        ),
        "negative_words_per_sentence": round(
            sum(map(sentence_length, negative)) / len(rows), 3
        ),
        "positive_optimism_markers": round(
            sum(len(POSITIVE.findall(text)) for text in positive) / len(rows), 3
        ),
        "negative_optimism_markers": round(
            sum(len(POSITIVE.findall(text)) for text in negative) / len(rows), 3
        ),
        "positive_cyrillic_rate": round(
            sum(bool(CYRILLIC.search(text)) for text in positive) / len(rows), 3
        ),
        "negative_cyrillic_rate": round(
            sum(bool(CYRILLIC.search(text)) for text in negative) / len(rows), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimism-source", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    feature_rows = {
        "russian_language": prepare_russian(args.output, args.count),
        "optimism": prepare_optimism(args.optimism_source, args.count),
        "atomic_sentences": prepare_atomic(args.count),
    }
    for feature, rows in feature_rows.items():
        write_jsonl(args.output / "data" / f"{feature}_pairs.jsonl", rows)
    rng = random.Random(20260802)
    report = {
        "requested_pairs": args.count,
        "features": {
            feature: summarize(rows) for feature, rows in feature_rows.items()
        },
        "manual_review": {
            feature: rng.sample(rows, min(5, len(rows)))
            for feature, rows in feature_rows.items()
        },
    }
    (args.output / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["features"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
