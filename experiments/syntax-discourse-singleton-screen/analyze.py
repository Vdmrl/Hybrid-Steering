"""Deterministic proxy summary for the three exploratory directions."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CONNECTIVES = re.compile(
    r"\b(therefore|however|because|first|second|finally|thus|consequently|"
    r"nevertheless|moreover|in contrast|as a result)\b",
    re.IGNORECASE,
)
CJK = re.compile(r"[\u3400-\u9fff]")


def rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metrics(text: str) -> dict[str, float | bool]:
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    return {
        "words_per_sentence": len(words) / max(len(sentences), 1),
        "connectives_per_100_words": 100 * len(CONNECTIVES.findall(text))
        / max(len(words), 1),
        "questions_per_response": text.count("?"),
        "has_cjk": bool(CJK.search(text)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for feature_dir in sorted(path for path in args.output.iterdir() if path.is_dir()):
        files = sorted(feature_dir.glob("screen-*.jsonl"))
        if not files:
            continue
        grouped: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
        for path in files:
            for row in rows(path):
                grouped[str(row["condition"])].append(metrics(str(row["response"])))
        feature = {}
        for condition, values in grouped.items():
            feature[condition] = {
                key: (
                    any(bool(value[key]) for value in values)
                    if key == "has_cjk"
                    else round(
                        sum(float(value[key]) for value in values) / len(values), 3
                    )
                )
                for key in values[0]
            }
        summary[feature_dir.name] = feature
    (args.output / "proxy-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
