"""Prepare non-synthetic humorous and numbered-list donor pairs.

This script deliberately only prepares and validates data.  Direction
extraction and GPU generation are separate explicit actions, so a bad source
cannot silently spend a GPU or Judge budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

HUMICROEDIT_URL = (
    "https://zenodo.org/records/3969509/files/"
    "semeval-2020-task-7-dataset.zip?download=1"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def find_column(row: dict[str, str], *names: str) -> str:
    lowered = {key.lower().replace("_", ""): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower().replace("_", ""))
        if value:
            return value.strip()
    raise ValueError(f"none of columns {names!r} found in {list(row)}")


def headline_from_edit(original: str, edit: str) -> str:
    """Replace Humicroedit's <.../> marker with the edited word or phrase."""
    return re.sub(r"<[^>]*?/>", edit.strip(), original, count=1).strip()


def prepare_humor(source_dir: Path, count: int) -> list[dict[str, Any]]:
    archive = source_dir / "humicroedit.zip"
    if not archive.exists():
        urllib.request.urlretrieve(HUMICROEDIT_URL, archive)
    extracted = source_dir / "humicroedit"
    if not extracted.exists():
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(extracted)
    candidates = list(extracted.rglob("train.csv")) + list(extracted.rglob("dev.csv"))
    if not candidates:
        raise FileNotFoundError("Humicroedit train/dev CSV was not found")

    best: dict[str, tuple[float, str, str]] = {}
    for path in candidates:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    original = find_column(row, "original")
                    edit = find_column(row, "edit")
                    grade = float(find_column(row, "meanGrade", "mean_grade"))
                except (ValueError, TypeError):
                    continue
                positive = headline_from_edit(original, edit)
                if not positive or positive == original or len(original.split()) < 4:
                    continue
                prior = best.get(original)
                if prior is None or grade > prior[0]:
                    best[original] = (grade, positive, path.name)

    selected = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:count]
    if len(selected) < count:
        raise RuntimeError(f"Humicroedit supplied only {len(selected)} usable pairs")
    return [
        {
            "source_id": f"humicroedit:{index}",
            "negative_text": original,
            "positive_text": positive,
            "human_funniness": round(grade, 4),
            "source_split": split,
        }
        for index, (original, (grade, positive, split)) in enumerate(selected)
    ]


def clean_numbered(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    return re.sub(r"\s+", " ", text)


def numbered_items(text: str) -> list[str]:
    items = [
        clean_numbered(item)
        for item in re.split(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)
        if clean_numbered(item)
    ]
    return items


def prepare_numbered(source: Path, count: int) -> list[dict[str, Any]]:
    text = source.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = rows.get("pairs", [rows])
    pairs = []
    for index, row in enumerate(rows):
        positive = str(row.get("positive_text", row.get("target", ""))).strip()
        negative = str(row.get("negative_text", row.get("source", ""))).strip()
        items = numbered_items(positive)
        if len(items) < 3 or not negative:
            continue
        items = items[:6]
        numbered = "\n".join(
            f"{number}. {item}" for number, item in enumerate(items, 1)
        )
        prose = " ".join(items)
        pairs.append(
            {
                "source_id": str(row.get("source_id", f"bullets:{index}")),
                "negative_text": prose,
                "positive_text": numbered,
                "item_count": len(items),
            }
        )
        if len(pairs) == count:
            break
    if len(pairs) < count:
        raise RuntimeError(
            f"bullet source supplied only {len(pairs)} clean numbered pairs"
        )
    return pairs


def validate(pairs: list[dict[str, Any]], numbered: bool = False) -> None:
    seen = set()
    for pair in pairs:
        negative, positive = pair["negative_text"], pair["positive_text"]
        if not negative or not positive or negative == positive:
            raise ValueError(f"invalid pair {pair['source_id']}")
        if positive in seen:
            raise ValueError(f"duplicate positive text: {pair['source_id']}")
        seen.add(positive)
        ratio = len(positive.split()) / max(1, len(negative.split()))
        if not 0.5 <= ratio <= 1.8:
            raise ValueError(f"length-ratio outlier {pair['source_id']}: {ratio:.2f}")
        if numbered and len(re.findall(r"(?m)^\d+\. ", positive)) < 3:
            raise ValueError(f"not a numbered list: {pair['source_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bullet-source", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=128)
    args = parser.parse_args()

    source_dir = args.output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    humor = prepare_humor(source_dir, args.pairs)
    numbered = prepare_numbered(args.bullet_source, args.pairs)
    validate(humor)
    validate(numbered, numbered=True)
    data_dir = args.output_dir / "data"
    write_json(data_dir / "humorous_pairs.json", {"pairs": humor})
    write_json(data_dir / "numbered_list_pairs.json", {"pairs": numbered})
    write_json(
        args.output_dir / "pair_manifest.json",
        {
            "humorous": {
                "source": "Humicroedit / SemEval-2020 Task 7",
                "n": len(humor),
            },
            "numbered_list": {"source": str(args.bullet_source), "n": len(numbered)},
        },
    )


if __name__ == "__main__":
    main()
