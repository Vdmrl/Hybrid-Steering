"""Resolve verified concept sources from the host-specific data root."""

import hashlib
import json
import re
from pathlib import Path


from hybrid_steering.paths import ROOT

CATALOG = ROOT / "concepts"


def load(concept: str, model: str) -> dict:
    if not all(re.fullmatch(r"[a-z0-9][a-z0-9.-]*", value) for value in (concept, model)):
        raise ValueError("Concept and model must be simple lowercase names")
    return json.loads((CATALOG / concept / f"{model}.json").read_text())


def asset_path(source: dict, asset: str, data_root: Path) -> Path:
    root = data_root.resolve()
    directory = (root / source["data_directory"]).resolve()
    path = (directory / source["assets"][asset]["path"]).resolve()
    if not directory.is_relative_to(root) or not path.is_relative_to(directory):
        raise ValueError(f"Asset escapes data root: {asset}")
    return path


def verify(source: dict, data_root: Path) -> None:
    for name, info in source["assets"].items():
        path = asset_path(source, name, data_root)
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != info["sha256"]:
            raise ValueError(f"SHA256 mismatch for {name}: {path}")
