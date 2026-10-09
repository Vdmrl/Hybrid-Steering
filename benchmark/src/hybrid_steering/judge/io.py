"""Small JSONL and provenance helpers shared by the Judge package."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, value: list | dict) -> None:
    """Replace a JSON or JSONL file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value)
    else:
        payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)
