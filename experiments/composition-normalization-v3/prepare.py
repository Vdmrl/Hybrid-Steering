"""Select deterministic feature-neutral dev/test prompts from feature_stories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

BASE_URL = (
    "https://huggingface.co/datasets/AntonKorznikov/feature_stories/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/{shard:04d}.parquet"
)
EXCLUDED = (
    "sycoph",
    "candor",
    "concrete",
    "abstract",
    "optim",
    "pessim",
    "joy",
    "sadness",
    "casual",
    "formal",
    "french",
)


def candidates(shard: int) -> list[dict]:
    with fsspec.open(BASE_URL.format(shard=shard), "rb").open() as stream:
        table = pq.ParquetFile(stream).read(
            columns=["id", "shared_setup", "class_name", "language", "genre", "model"]
        )
    rows = []
    for row in table.to_pylist():
        label = (row["class_name"] or "").lower()
        setup = (row["shared_setup"] or "").strip()
        if (
            row["language"] == "English"
            and 80 <= len(setup) <= 1200
            and not any(term in label for term in EXCLUDED)
        ):
            rows.append(row)
    return rows


def write(path: Path, rows: list[dict], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "prompt": (
                        f"{row['shared_setup']}\n\n"
                        "What should the person say or do next? "
                        "Answer directly and explain briefly."
                    ),
                    "split": split,
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev", type=int, default=32)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    dev_path, test_path = args.output_dir / "dev.jsonl", args.output_dir / "test.jsonl"
    if dev_path.exists() and test_path.exists():
        print(f"reuse prompts: {dev_path} {test_path}")
        return

    unique = {}
    for shard in range(13):
        for row in candidates(shard):
            unique.setdefault(row["shared_setup"], row)
        if len(unique) >= 4 * (args.dev + args.test):
            break
    rows = list(unique.values())
    random.Random(args.seed).shuffle(rows)
    selected = rows[: args.dev + args.test]
    if len(selected) != args.dev + args.test:
        raise RuntimeError(
            f"need {args.dev + args.test} prompts, found {len(selected)}"
        )
    write(dev_path, selected[: args.dev], "dev")
    write(test_path, selected[args.dev :], "test")
    print(f"prepared dev={args.dev} test={args.test}")


if __name__ == "__main__":
    main()
