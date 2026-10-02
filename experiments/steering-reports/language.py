import argparse
import random
import re
from pathlib import Path
from typing import Sequence
from typing import TypedDict

from datasets import Dataset, load_dataset

LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Russian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "it": "Italian",
    "pt": "Portuguese",
}

# Keep these in ONE fixed control language.
# The only semantic variable should be the requested output language.
TRAIN_TEMPLATES = (
    "Respond only in {language}.",
    "Write your entire answer in {language}.",
    "Use {language} for your response.",
    "Answer exclusively in {language}.",
)

# Held-out wording: useful later for checking that you found language,
# rather than memorized instruction phrasing.
TEST_TEMPLATES = (
    "Your response must be written in {language}.",
    "Provide the answer entirely in {language}.",
)


class ContentExample(TypedDict):
    source_id: str
    context: str
    question: str


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_content_examples(n: int, seed: int = 42) -> list[ContentExample]:
    ds = load_dataset("rajpurkar/squad", split="train")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    examples = []

    for idx in indices:
        row = ds[idx]

        context = clean_text(row["context"])
        question = clean_text(row["question"])

        if len(context.split()) < 80:
            continue
        if len(question.split()) < 4:
            continue

        words = context.split()
        context = " ".join(words[:450])

        example: ContentExample = {
            "source_id": str(row["id"]),
            "context": context,
            "question": question,
        }
        examples.append(example)

        if len(examples) == n:
            break

    if len(examples) < n:
        raise RuntimeError(f"Could only find {len(examples)} suitable examples.")

    return examples


def build_prompt(
    content: str,
    question: str,
    language: str,
    template: str,
) -> str:
    return f"{template.format(language=language)}\n\nContext:\n{content}\n\nQuestion:\n{question}\n\nAnswer:"


def build_row(
    pair_id: int,
    condition: str,
    language_code: str,
    split: str,
    template: str,
    example: ContentExample,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "condition": condition,
        "split": split,
        "language_code": language_code,
        "prompt": build_prompt(
            content=example["context"],
            question=example["question"],
            language=LANGUAGE_NAMES[language_code],
            template=template,
        ),
    }


def generate_paired_language_dataset(
    lang_a: str = "en",
    lang_b: str = "ru",
    n_pairs: int = 200,
    seed: int = 42,
    heldout_fraction: float = 0.25,
) -> Dataset:
    if lang_a not in LANGUAGE_NAMES:
        raise ValueError(f"Unknown language code: {lang_a}")
    if lang_b not in LANGUAGE_NAMES:
        raise ValueError(f"Unknown language code: {lang_b}")

    rng = random.Random(seed)

    contents = make_content_examples(n_pairs, seed=seed)

    n_heldout = round(n_pairs * heldout_fraction)
    heldout_ids = set(rng.sample(range(n_pairs), n_heldout))

    rows: list[dict[str, object]] = []

    for pair_id, example in enumerate(contents):
        split = "eval" if pair_id in heldout_ids else "discovery"
        templates = TEST_TEMPLATES if split == "eval" else TRAIN_TEMPLATES
        template_id = rng.randrange(len(templates))
        template = templates[template_id]

        for condition, language_code in (("A", lang_a), ("B", lang_b)):
            rows.append(
                build_row(
                    pair_id=pair_id,
                    condition=condition,
                    language_code=language_code,
                    split=split,
                    template=template,
                    example=example,
                )
            )

    return Dataset.from_list(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paired language-control prompts from SQuAD.")
    parser.add_argument(
        "--lang-a",
        choices=LANGUAGE_NAMES,
        default="en",
        help="Language code for condition A.",
    )
    parser.add_argument(
        "--lang-b",
        choices=LANGUAGE_NAMES,
        default="ru",
        help="Language code for condition B.",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=200,
        help="Number of paired examples to generate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--heldout-fraction",
        type=float,
        default=0.25,
        help="Fraction of pairs using held-out instruction templates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paired_language_200"),
        help="Directory for the saved Hugging Face dataset.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Optional path for a JSONL export.",
    )
    args = parser.parse_args(argv)

    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    if not 0 <= args.heldout_fraction <= 1:
        parser.error("--heldout-fraction must be between 0 and 1")

    ds = generate_paired_language_dataset(
        lang_a=args.lang_a,
        lang_b=args.lang_b,
        n_pairs=args.pairs,
        seed=args.seed,
        heldout_fraction=args.heldout_fraction,
    )
    ds.save_to_disk(args.output_dir)

    if args.jsonl:
        ds.to_json(args.jsonl, orient="records", lines=True)

    print(f"Wrote {len(ds)} rows to {args.output_dir}")
    if args.jsonl:
        print(f"Wrote JSONL to {args.jsonl}")


if __name__ == "__main__":
    main()
