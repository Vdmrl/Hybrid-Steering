"""Build 64 multilingual source-to-French matched donor pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from langdetect import DetectorFactory, detect
from openai import OpenAI

DetectorFactory.seed = 0
LANGUAGES = (
    "English",
    "German",
    "Spanish",
    "Russian",
    "Mandarin Chinese",
)
QUOTAS = {
    "English": 13,
    "German": 13,
    "Spanish": 13,
    "Russian": 13,
    "Mandarin Chinese": 12,
}
SYSTEM = """Translate the supplied text into natural French.
Preserve its meaning, factual claims, emotional stance, register, and paragraph
structure. Do not summarize, explain, censor, or add content. Return only the
French translation without markdown."""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join([str(seed), *parts]).encode()).hexdigest()


def select_rows(parquet: Path, seed: int) -> list[dict[str, Any]]:
    columns = [
        "id",
        "concept",
        "antagonist",
        "concept_text",
        "antagonist_text",
        "class_name",
        "language",
        "genre",
        "model",
        "shared_setup",
    ]
    rows = pq.read_table(parquet, columns=columns).to_pylist()
    selected: list[dict[str, Any]] = []
    used_axes: set[tuple[str, str]] = set()
    for language in LANGUAGES:
        candidates = sorted(
            (row for row in rows if row["language"] == language),
            key=lambda row: stable_key(seed, language, str(row["id"])),
        )
        for row in candidates:
            axis = (str(row["concept"]), str(row["antagonist"]))
            if axis in used_axes:
                continue
            side = (
                "concept_text"
                if int(stable_key(seed, str(row["id"]))[:2], 16) % 2 == 0
                else "antagonist_text"
            )
            selected.append(
                {
                    "id": f"french-{row['id']}",
                    "source_id": row["id"],
                    "source_language": language,
                    "source_side": side,
                    "source_text": row[side],
                    "negative_text": row[side],
                    "concept": row["concept"],
                    "antagonist": row["antagonist"],
                    "class_name": row["class_name"],
                    "genre": row["genre"],
                    "generator_model": row["model"],
                    "shared_setup": row["shared_setup"],
                }
            )
            used_axes.add(axis)
            if (
                sum(item["source_language"] == language for item in selected)
                == QUOTAS[language]
            ):
                break
    counts = {
        language: sum(row["source_language"] == language for row in selected)
        for language in LANGUAGES
    }
    if counts != QUOTAS:
        raise RuntimeError(f"could not satisfy language quotas: {counts}")
    if len({(row["concept"], row["antagonist"]) for row in selected}) != 64:
        raise RuntimeError("donors must cover 64 distinct concept axes")
    return selected


def existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        row["id"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def translate(client: OpenAI, model: str, row: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=1400,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Source language: {row['source_language']}\n\n"
                            f"{row['source_text']}"
                        ),
                    },
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            if len(text) < 40 or detect(text) != "fr":
                raise ValueError("translation is empty, too short, or not French")
            usage = response.usage
            return {
                **row,
                "positive_text": text,
                "translation_model": model,
                "translation_input_tokens": int(usage.prompt_tokens if usage else 0),
                "translation_output_tokens": int(
                    usage.completion_tokens if usage else 0
                ),
            }
        except Exception as error:  # noqa: BLE001 - retry provider/schema failures
            last_error = error
    raise RuntimeError(f"{row['id']}: translation failed") from last_error


def self_test() -> None:
    assert sum(QUOTAS.values()) == 64
    assert set(QUOTAS) == set(LANGUAGES)
    assert stable_key(1, "x") == stable_key(1, "x")
    print("prepare self-test passed")


def main() -> None:
    args = arguments()
    if args.self_test:
        self_test()
        return
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    selected = select_rows(args.parquet, args.seed)
    done = existing_rows(args.output)
    pending = [row for row in selected if row["id"] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=240,
        max_retries=4,
    )
    with (
        args.output.open("a", encoding="utf-8") as stream,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {
            pool.submit(translate, client, args.model, row): row for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            done[result["id"]] = result
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"translated {index}/{len(pending)}", flush=True)

    ordered = [done[row["id"]] for row in selected]
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pairs": len(ordered),
                "source_languages": {
                    language: sum(row["source_language"] == language for row in ordered)
                    for language in LANGUAGES
                },
                "distinct_axes": len(
                    {(row["concept"], row["antagonist"]) for row in ordered}
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
