from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def paired_layout(text: str) -> tuple[str, str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        if sentence.strip()
    ][:8]
    if len(sentences) < 3:
        raise ValueError("need at least three sentences")
    return "\n".join(f"- {sentence}" for sentence in sentences), " ".join(sentences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
    ]
    result = []
    for row in rows:
        try:
            positive, negative = paired_layout(row["positive_text"])
        except ValueError:
            continue
        result.append(
            {
                "id": f"bulleted:{row['id']}",
                "positive_text": positive,
                "negative_text": negative,
                "source_id": row["id"],
            }
        )
        if len(result) == args.count:
            break
    if len(result) != args.count:
        raise RuntimeError(f"need {args.count} bullet pairs, found {len(result)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result),
        encoding="utf-8",
    )
    print(f"wrote {len(result)} bullet/prose pairs to {args.output}")


if __name__ == "__main__":
    main()
