from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import requests


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--shards", default="10,9,11")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    rows = []
    for shard in (int(value) for value in args.shards.split(",")):
        url = (
            "https://huggingface.co/datasets/AntonKorznikov/"
            "feature_stories/resolve/main/data/"
            f"train-{shard:05d}-of-00016.parquet"
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
            handle.flush()
            table = pq.read_table(
                handle.name,
                filters=[
                    ("concept", "=", "uses first-person"),
                    ("antagonist", "=", "uses third-person"),
                    ("language", "=", "English"),
                ],
                columns=[
                    "id",
                    "concept_text",
                    "antagonist_text",
                    "genre",
                    "model",
                ],
            )
            rows.extend(table.to_pylist())
        if len(rows) >= args.count:
            break
    rows = sorted(rows, key=lambda row: (row["model"], row["genre"], row["id"]))[
        : args.count
    ]
    if len(rows) != args.count:
        raise RuntimeError(f"need {args.count} pairs, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "positive_text": row["concept_text"],
                        "negative_text": row["antagonist_text"],
                        "genre": row["genre"],
                        "generator": row["model"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(rows)} first-person pairs to {args.output}")


if __name__ == "__main__":
    main()
