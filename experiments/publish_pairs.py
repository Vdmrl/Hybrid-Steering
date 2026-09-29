"""Publish aligned concept pairs to the Hub dataset.

``--concept en-ru`` selects OPUS-100 pairs. ``--jsonl`` publishes pairs that
are already prepared, which is the path for a concept that is not a language.
``positive_text`` is the target and ``negative_text`` is the source.

OPUS rows are the first ``--pairs`` train rows after ``shuffle(seed=42)`` with
more than 20 words on each side, letters only from that side's script, and at
least one letter that is not all uppercase.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from hybrid_steering.direction import CONCEPT_DATASET, read_pairs
from hybrid_steering.runtime import write_jsonl

DATASET = CONCEPT_DATASET
SEED = 42
MIN_WORDS = 20
ALPHABETS = {
    "en": frozenset("abcdefghijklmnopqrstuvwxyz"),
    "ru": frozenset("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    "fr": frozenset("abcdefghijklmnopqrstuvwxyzàâæçéèêëîïôœùûüÿ"),
}


def has_only_language_letters(text: str, language: str) -> bool:
    alphabet = ALPHABETS[language]
    return all(not character.isalpha() or character.lower() in alphabet for character in text)


def has_letters_and_is_not_uppercase(text: str) -> bool:
    return any(character.isalpha() for character in text) and not text.isupper()


def opus_pairs(source: str, target: str, pairs: int) -> list[dict[str, str]]:
    config = "-".join(sorted((source, target)))
    dataset = load_dataset("Helsinki-NLP/opus-100", config, split="train").shuffle(seed=SEED)
    selected: list[dict[str, str]] = []
    for index, row in enumerate(dataset):
        translation = row["translation"]
        negative, positive = translation[source].strip(), translation[target].strip()
        if min(len(negative.split()), len(positive.split())) <= MIN_WORDS:
            continue
        if not (
            has_only_language_letters(negative, source)
            and has_only_language_letters(positive, target)
            and has_letters_and_is_not_uppercase(negative)
            and has_letters_and_is_not_uppercase(positive)
        ):
            continue
        selected.append(
            {
                "pair_id": f"opus-{source}-{target}-{index}",
                "split": "train",
                "negative_text": negative,
                "positive_text": positive,
                "source_id": f"opus-100:{config}:{index}",
                "source_license": "cc-by-4.0",
            }
        )
        if len(selected) == pairs:
            return selected
    raise RuntimeError(f"found only {len(selected)} usable {source}->{target} pairs")


def upload(concept: str, name: str, rows: list[dict[str, str]]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / name
        write_jsonl(path, rows)
        HfApi().upload_file(
            path_or_fileobj=path,
            path_in_repo=f"concepts/{concept}/data/{name}",
            repo_id=DATASET,
            repo_type="dataset",
            commit_message=f"Publish {concept} {name}",
        )
    print(f"uploaded {len(rows)} rows to {DATASET} concepts/{concept}/data/{name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", required=True, help="dataset directory, such as en-ru")
    parser.add_argument("--jsonl", type=Path, help="prepared pairs; skips OPUS selection")
    parser.add_argument("--pairs", type=int, default=2500)
    args = parser.parse_args()
    concept = args.concept.strip().replace("->", "-")
    if args.jsonl:
        rows = read_pairs(args.jsonl)
    else:
        source, target = concept.split("-")
        rows = opus_pairs(source, target, args.pairs)
    upload(concept, "pairs.jsonl", rows)


if __name__ == "__main__":
    main()
