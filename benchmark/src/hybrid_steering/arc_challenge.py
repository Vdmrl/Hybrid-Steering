"""ARC-Challenge validation: pinned JSONL snapshot and zero-shot choice scoring.

This is a generative exact-choice protocol, not leaderboard likelihood scoring.
The 299-row snapshot is downloaded once; benchmark plans pin its SHA256.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET = "allenai/ai2_arc"
CONFIG = "ARC-Challenge"
SPLIT = "validation"
EXPECTED = 299
API = "https://datasets-server.huggingface.co/rows"
REVISION_API = "https://huggingface.co/api/datasets/allenai/ai2_arc"
PROMPT_TEMPLATE = "Answer this multiple-choice science question. Return only the label of the correct option.\n\nQuestion: {question}\n{choices}\nAnswer:"


def validate_row(row: dict) -> dict:
    """Retain only the fields needed to reproduce one question and its answer."""
    options = row["choices"]
    labels, texts = options["label"], options["text"]
    if (not isinstance(row["id"], str) or not isinstance(row["question"], str)
            or not 3 <= len(labels) == len(texts) <= 5
            or not all(isinstance(label, str) and re.fullmatch(r"[A-E1-5]", label) for label in labels)
            or not all(isinstance(text, str) and text.strip() for text in texts)
            or len(set(labels)) != len(labels) or row["answerKey"] not in labels):
        raise ValueError("Invalid ARC-Challenge row")
    return {"id": row["id"], "question": row["question"],
            "choices": {"label": labels, "text": texts}, "answerKey": row["answerKey"]}


def prompt(row: dict) -> str:
    choices = "\n".join(f"{label}. {text}" for label, text in
                        zip(row["choices"]["label"], row["choices"]["text"], strict=True))
    return PROMPT_TEMPLATE.format(question=row["question"], choices=choices)


def score(response: str, row: dict) -> dict:
    """Count a single explicit label; separately record strict label-only format."""
    labels = row["choices"]["label"]
    allowed = "|".join(re.escape(label) for label in labels)
    strict = re.fullmatch(rf"\s*({allowed})[.)]?\s*", response, re.IGNORECASE)
    relaxed = strict or re.fullmatch(
        rf"\s*(?:the\s+)?(?:correct\s+)?(?:answer|option)\s*(?:is|:)\s*\(?({allowed})\)?[.!]?\s*",
        response, re.IGNORECASE,
    )
    predicted = relaxed.group(1).upper() if relaxed else None
    return {"arc_predicted": predicted, "arc_correct": predicted == row["answerKey"],
            "arc_format_valid": bool(strict)}


def _json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def download(path: Path) -> dict:
    """Download the public 299-row validation split and record exact provenance."""
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {path}")
    revision = _json(REVISION_API)["sha"]
    rows = []
    for offset in range(0, EXPECTED, 100):
        query = urlencode({"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                           "offset": offset, "length": min(100, EXPECTED - offset)})
        page = _json(f"{API}?{query}")
        if page["num_rows_total"] != EXPECTED or len(page["rows"]) != min(100, EXPECTED - offset):
            raise ValueError("ARC-Challenge page size changed")
        for index, item in enumerate(page["rows"], offset):
            if item["row_idx"] != index or item.get("truncated_cells"):
                raise ValueError("ARC-Challenge viewer returned a truncated or reordered row")
            rows.append(validate_row(item["row"]))
    if _json(REVISION_API)["sha"] != revision or len({row["id"] for row in rows}) != EXPECTED:
        raise ValueError("ARC-Challenge revision changed or duplicate IDs found")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    manifest = {"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                "revision": revision, "rows": EXPECTED, "sha256": digest,
                "source": API, "protocol": "zero-shot-generative-choice-v1"}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(download(args.output), indent=2))
