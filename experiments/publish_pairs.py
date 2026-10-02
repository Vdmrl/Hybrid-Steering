"""Publish aligned concept pairs to the Hub dataset.

``--concept en-ru`` selects OPUS-100 pairs. ``--jsonl`` publishes pairs that
are already prepared, which is the path for a concept that is not a language.
``positive_text`` is the target and ``negative_text`` is the source.

OPUS rows are the first ``--pairs`` train rows after ``shuffle(seed=42)`` with
more than 20 words on each side, letters only from that side's script, and at
least one letter that is not all uppercase. Chinese is not space-separated, so
its length is the number of Han characters and the same threshold applies.
``--output`` writes that selection locally and does not upload it.
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


def letter_allowed(character: str, language: str) -> bool:
    code = ord(character)
    if language == "zh":
        return 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF
    if language == "ar":
        return (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        )
    if language == "hi":
        return 0x0900 <= code <= 0x097F
    return character.lower() in ALPHABETS[language]


def has_only_language_letters(text: str, language: str) -> bool:
    return all(not character.isalpha() or letter_allowed(character, language) for character in text)


def has_letters_and_is_not_uppercase(text: str) -> bool:
    return any(character.isalpha() for character in text) and not text.isupper()


def text_units(text: str, language: str) -> int:
    """Words, or Han characters when the language is Chinese."""
    if language == "zh":
        return sum(1 for character in text if letter_allowed(character, "zh"))
    return len(text.split())


def opus_pairs(source: str, target: str, pairs: int) -> list[dict[str, str]]:
    config = "-".join(sorted((source, target)))
    dataset = load_dataset("Helsinki-NLP/opus-100", config, split="train").shuffle(seed=SEED)
    selected: list[dict[str, str]] = []
    for index, row in enumerate(dataset):
        translation = row["translation"]
        negative, positive = translation[source].strip(), translation[target].strip()
        if min(text_units(negative, source), text_units(positive, target)) <= MIN_WORDS:
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
    parser.add_argument(
        "--concept", required=True, help="dataset directory, such as en-ru or en-zh"
    )
    parser.add_argument("--jsonl", type=Path, help="prepared pairs; skips OPUS selection")
    parser.add_argument("--pairs", type=int, default=2500)
    parser.add_argument(
        "--output", type=Path, help="write the selection locally and skip the upload"
    )
    args = parser.parse_args()
    concept = args.concept.strip().replace("->", "-")
    if args.jsonl:
        rows = read_pairs(args.jsonl)
    else:
        source, target = concept.split("-")
        rows = opus_pairs(source, target, args.pairs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output, rows)
        print(f"wrote {len(rows)} rows to {args.output}", flush=True)
        return
    upload(concept, "pairs.jsonl", rows)


if __name__ == "__main__":
    main()
