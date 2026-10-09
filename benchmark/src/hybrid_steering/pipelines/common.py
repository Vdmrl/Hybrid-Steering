"""Host-specific paths shared by the command-line pipelines."""

import os
from pathlib import Path


def configure_data_root(path: Path) -> Path:
    root = path.resolve()
    os.environ["GDN_DATA_ROOT"] = str(root)
    os.environ.setdefault("HF_HOME", str(root / "cache/huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "cache"))
    return root
