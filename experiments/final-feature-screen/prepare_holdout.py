"""Sample a fixed, new neutral holdout from the public Alpaca instruction data."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)
SEED = 20260802
BLOCKED = ("list", "step", "bullet", "write a", "generate", "program", "code")
PREFIXES = ("how ", "what ", "explain ", "describe ", "give ", "why ")


def eligible(instruction: str) -> bool:
    text = " ".join(instruction.lower().split())
    words = text.split()
    return (
        5 <= len(words) <= 32
        and text.startswith(PREFIXES)
        and not any(word in text for word in BLOCKED)
    )


def select(rows: list[dict], count: int) -> list[str]:
    candidates = list(
        dict.fromkeys(
            str(row.get("instruction", "")).strip()
            for row in rows
            if eligible(str(row.get("instruction", ""))) and not row.get("input")
        )
    )
    if len(candidates) < count:
        raise ValueError(f"need {count} eligible prompts, found {len(candidates)}")
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    return candidates[:count]


def download(path: Path) -> list[dict]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_holdout(source: Path, output: Path, count: int) -> None:
    prompts = select(download(source), count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps({"id": f"final-holdout-{index:03d}", "prompt": prompt}) + "\n"
            for index, prompt in enumerate(prompts)
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    write_holdout(args.source, args.output, args.count)
