"""Minimal singleton screen for three syntax/discourse directions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]
BASE = ROOT / "experiments" / "replacement-singleton-screen" / "run.py"
spec = importlib.util.spec_from_file_location("replacement_screen", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE}")
screen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = screen
spec.loader.exec_module(screen)

FEATURES = {
    "atomic_sentences": (
        "short, simple, atomic sentences; split combined claims into separate sentences",
        "syntactically complex sentences with clauses joined together",
    ),
    "explicit_connectives": (
        "explicit logical connectives such as therefore, however, because, first, and finally",
        "implicit transitions with the same reasoning but no explicit connective words",
    ),
    "rhetorical_questions": (
        "rhetorical questions woven naturally into the explanation",
        "fully declarative prose with no rhetorical questions",
    ),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    target, opposite = FEATURES[args.feature]
    frame = pd.read_parquet(args.source, columns=["input_text"])
    texts = [
        str(text).strip()
        for text in frame["input_text"].dropna().drop_duplicates()
        if 45 <= len(str(text).split()) <= 130
    ][: args.count]
    if len(texts) < args.count:
        raise RuntimeError(f"only {len(texts)}/{args.count} source texts")
    tokenizer, model = screen.runner.load_model(args.model)
    positive = screen.rewrite(model, tokenizer, texts, target, args.batch)
    negative = screen.rewrite(model, tokenizer, texts, opposite, args.batch)
    pairs = [
        {
            "source_id": f"synthetic:{index}",
            "positive_text": pos,
            "negative_text": neg,
            "length_ratio": round(len(pos.split()) / max(len(neg.split()), 1), 3),
        }
        for index, (pos, neg) in enumerate(zip(positive, negative, strict=True))
        if pos and neg and pos != neg
    ]
    if len(pairs) < args.count:
        raise RuntimeError(f"only {len(pairs)}/{args.count} usable pairs")
    write_json(
        args.output / "data" / f"{args.feature}_pairs.json",
        {
            "feature": args.feature,
            "target": target,
            "opposite": opposite,
            "pairs": pairs,
        },
    )


def delegate(args: argparse.Namespace, command: str) -> None:
    args.command = command
    if command == "build-direction":
        screen.build_direction(args)
    else:
        screen.screen(args)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "build-direction", "screen"))
    parser.add_argument("--feature", choices=FEATURES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--spec", action="append")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--tag", default="screen")
    parser.add_argument("--direction", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.command == "prepare":
        if args.source is None:
            raise ValueError("--source is required")
        prepare(args)
    elif args.command == "build-direction":
        delegate(args, "build-direction")
    else:
        if args.prompts is None or not args.spec:
            raise ValueError("--prompts and --spec are required")
        delegate(args, "screen")


if __name__ == "__main__":
    main()
