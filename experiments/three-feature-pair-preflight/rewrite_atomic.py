"""Create content-matched atomic/compound sentence pairs locally with Qwen."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).parents[2]
BASE = ROOT / "experiments" / "replacement-singleton-screen" / "run.py"
spec = importlib.util.spec_from_file_location("replacement_screen", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE}")
screen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = screen
spec.loader.exec_module(screen)

SYSTEM = (
    "Rewrite the input as exactly two short, complete, standalone sentences. "
    "Preserve every fact, named entity, number, meaning, tone, and approximate "
    "total length. Do not add headings, bullets, commentary, or new facts. "
    "Return only the rewritten text."
)


def rewrite(model: Any, tokenizer: Any, texts: list[str], batch_size: int) -> list[str]:
    result = []
    for start in range(0, len(texts), batch_size):
        inputs = screen.batched_chat(
            tokenizer, texts[start : start + batch_size], SYSTEM
        )
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=160,
            pad_token_id=tokenizer.eos_token_id,
        )
        width = inputs["input_ids"].shape[1]
        result.extend(
            text.strip()
            for text in tokenizer.batch_decode(
                outputs[:, width:], skip_special_tokens=True
            )
        )
        print(f"rewritten {min(start + batch_size, len(texts))}/{len(texts)}")
    return result


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text) if part.strip()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=144)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()
    frame = pd.read_parquet(args.source)
    english = []
    for value in frame["translation"]:
        text = str(value["en"]).strip()
        if (
            18 <= len(text.split()) <= 60
            and sentence_count(text) == 1
            and text not in english
        ):
            english.append(text)
        if len(english) == args.count:
            break
    if len(english) < args.count:
        raise RuntimeError(f"only {len(english)}/{args.count} English sources")
    tokenizer, model = screen.runner.load_model(args.model)
    atomic = rewrite(model, tokenizer, english, args.batch)
    rows = []
    for index, (source, target) in enumerate(zip(english, atomic, strict=True)):
        length_ratio = len(target.split()) / max(len(source.split()), 1)
        if sentence_count(target) != 2 or not 0.75 <= length_ratio <= 1.3:
            continue
        rows.append(
            {
                "source_id": f"opus100-english:qwen-atomic:{index}",
                "positive_text": target,
                "negative_text": source,
            }
        )
    if len(rows) < 128:
        raise RuntimeError(f"only {len(rows)}/128 clean atomic rewrites")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
