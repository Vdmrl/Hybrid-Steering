"""Shared rubric and stable task IDs for the current Judge protocol."""

import hashlib
import json
from pathlib import Path

from .io import digest, now, read_jsonl, write
from hybrid_steering.paths import ROOT


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def render(concept: str, rubric: str = "judge_v4.md") -> str:
    """Insert concept-specific guidance into the shared scoring rubric."""
    if not concept or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in concept):
        raise ValueError("Concept must be a simple lowercase name")

    guide_path = ROOT / "prompts/concepts" / f"{concept}.md"
    guide = guide_path.read_text().strip()
    template = (ROOT / "prompts" / rubric).read_text()
    return template.replace("{{concept}}", guide)


def task_id(row: dict, protocol: dict) -> str:
    """The same text and rubric get the same ID across experiment conditions."""
    return fingerprint({
        "protocol": protocol,
        "prompt": row["prompt"],
        "response": row["response"],
    })
