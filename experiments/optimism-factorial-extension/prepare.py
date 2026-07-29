"""Select balanced optimism/pessimism pairs without downloading the dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

BASE_URL = (
    "https://huggingface.co/datasets/AntonKorznikov/feature_stories/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/{shard:04d}.parquet"
)
CLASS_NAME = "Optimism vs Pessimism as Basic Disposition (selected)"


def matching_rows(shard: int) -> list[dict]:
    with fsspec.open(
        BASE_URL.format(shard=shard),
        "rb",
        block_size=1 << 20,
        cache_type="readahead",
    ).open() as stream:
        parquet = pq.ParquetFile(stream)
        class_index = parquet.schema.names.index("class_name")
        groups = []
        for index in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(index).column(class_index).statistics
            if stats.min <= CLASS_NAME <= stats.max:
                groups.append(index)
        table = parquet.read_row_groups(groups)
    return [
        row
        for row in table.to_pylist()
        if row["class_name"] == CLASS_NAME and row["language"] == "English"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=104)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    if args.output.exists() and sum(
        bool(line.strip())
        for line in args.output.read_text(encoding="utf-8").splitlines()
    ) >= args.count:
        print(f"reuse optimism pairs: {args.output}")
        return

    candidates = matching_rows(4) + matching_rows(12)
    unique = {row["id"]: row for row in candidates}
    by_concept: dict[str, list[dict]] = defaultdict(list)
    for row in unique.values():
        by_concept[row["concept"]].append(row)
    if len(by_concept) != 4 or args.count % len(by_concept):
        raise RuntimeError("expected four balanced optimism sub-concepts")

    rng = random.Random(args.seed)
    selected = []
    per_concept = args.count // len(by_concept)
    for concept in sorted(by_concept):
        rows = by_concept[concept]
        rng.shuffle(rows)
        selected.extend(rows[:per_concept])
    rng.shuffle(selected)
    if len(selected) != args.count:
        raise RuntimeError("not enough optimism rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(
                {
                    "id": row["id"],
                    "positive_text": row["concept_text"],
                    "negative_text": row["antagonist_text"],
                    "concept": row["concept"],
                    "antagonist": row["antagonist"],
                    "shared_setup": row["shared_setup"],
                    "language": row["language"],
                    "genre": row["genre"],
                    "model": row["model"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    print(f"selected={len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
