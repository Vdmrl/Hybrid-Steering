"""Select deterministic English joy/sadness pairs from feature_stories."""

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


def matching_rows(shard: int) -> list[dict]:
    with fsspec.open(
        BASE_URL.format(shard=shard),
        "rb",
        block_size=1 << 20,
        cache_type="readahead",
    ).open() as stream:
        parquet = pq.ParquetFile(stream)
        concept_index = parquet.schema.names.index("concept")
        groups = []
        for index in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(index).column(concept_index).statistics
            if stats and stats.min <= "joy" <= stats.max:
                groups.append(index)
        if not groups:
            return []
        table = parquet.read_row_groups(groups)
    return [
        row
        for row in table.to_pylist()
        if row["language"] == "English"
        and row["concept"].strip().lower() == "joy"
        and row["antagonist"].strip().lower() == "sadness"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    if args.output.exists():
        existing = [
            line
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(existing) >= args.count:
            print(f"reuse joy pairs: {args.output} ({len(existing)})")
            return

    rows = []
    for shard in range(13):
        found = matching_rows(shard)
        rows.extend(found)
        print(f"joy scan: shard={shard:04d} found={len(found)} total={len(rows)}")

    unique = {row["id"]: row for row in rows}
    candidates = list(unique.values())
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.count]
    if len(selected) < args.count:
        raise RuntimeError(f"need {args.count} joy rows, found {len(selected)}")

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
                    "narrative_guidance": row["narrative_guidance"],
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
