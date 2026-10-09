"""Resolve project files after the one-package migration."""

from pathlib import Path

def _repository_root() -> Path:
    candidates = (Path(__file__).resolve().parents[2], Path.cwd(), *Path.cwd().parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "concepts").is_dir():
            return candidate
    raise FileNotFoundError("Run from the repository or install in editable mode")


ROOT = _repository_root()


def project_file(name: str) -> Path:
    """Find a code/config file; preserve old `configs/X` plan references."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Project path must be relative: {name}")
    direct = ROOT / relative
    if direct.is_file():
        return direct
    if relative.parts[:1] == ("configs",) and len(relative.parts) == 2:
        folder = ROOT / "data/legacy-config/configs"
        matches = [folder / model / relative.name for model in ("4b", "9b", "shared")]
        matches = [path for path in matches if path.is_file()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous historical config: {name}")
    raise FileNotFoundError(f"Project file not found: {name}")
