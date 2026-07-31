import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("build.py")
SPEC = importlib.util.spec_from_file_location("dashboard3_real_build", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_uses_real_exp3_summary() -> None:
    root = Path(__file__).parent
    summary = json.loads((root / "exp3_summary.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "exp3_selection.json").read_text(encoding="utf-8"))
    html = MODULE.build(summary, selection)
    assert "composition-normalization-v3" in html
    assert "gdn_rss_r4_1111" in html
    assert "GDN raw, rank 1" in html
    assert "Тройки: сила и сохранение качества" in html
    assert "__DATA__" not in html
