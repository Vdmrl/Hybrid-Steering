import json
from pathlib import Path

from build import build


def test_build_uses_real_exp3_summary() -> None:
    root = Path(__file__).parent
    summary = json.loads((root / "exp3_summary.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "exp3_selection.json").read_text(encoding="utf-8"))
    html = build(summary, selection)
    assert "composition-normalization-v3" in html
    assert "gdn_rss_r4_1111" in html
    assert "GDN raw, rank 1" in html
    assert "__DATA__" not in html
