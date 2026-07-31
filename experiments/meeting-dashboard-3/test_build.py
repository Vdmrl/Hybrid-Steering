import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).with_name("build.py")
SPEC = importlib.util.spec_from_file_location("meeting_dashboard_3", MODULE)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_dashboard_builds_from_committed_summaries():
    page = BUILD.build(
        json.loads(BUILD.DEFAULT_SUMMARY.read_text(encoding="utf-8")),
        json.loads(BUILD.DEFAULT_COMPARISONS.read_text(encoding="utf-8")),
        json.loads(BUILD.DEFAULT_ANALYSIS.read_text(encoding="utf-8")),
        json.loads(BUILD.DEFAULT_EXP2.read_text(encoding="utf-8")),
    )

    assert "Что происходит, когда мы складываем несколько концептов?" in page
    assert "RSS-нормализация" in page
    assert "Короткий ответ" in page
    assert "__DATA__" not in page
