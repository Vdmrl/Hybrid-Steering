import importlib.util
from pathlib import Path

MODULE = Path(__file__).with_name("build.py")
SPEC = importlib.util.spec_from_file_location("composition_dashboard", MODULE)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_dashboard_builds_from_committed_results():
    import json

    summary = json.loads(BUILD.DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    comparisons = json.loads(BUILD.DEFAULT_COMPARISONS.read_text(encoding="utf-8"))
    analysis = json.loads(BUILD.DEFAULT_ANALYSIS.read_text(encoding="utf-8"))
    page = BUILD.build(summary, comparisons, analysis)

    assert "Composition analysis dashboard" in page
    assert "Does extra SVD rank help?" in page
    assert "Classical activation steering" in page
    assert "Fair head-to-head on the same 96 held-out prompts" in page
    assert "Does rank-1 compose?" in page
    assert "Next confirmatory experiment" in page
    assert "__DATA__" not in page
