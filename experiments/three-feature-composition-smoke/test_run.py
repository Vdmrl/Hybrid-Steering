from pathlib import Path


def test_smoke_contains_only_baseline_and_triple():
    text = Path(__file__).with_name("run.py").read_text(encoding="utf-8")
    assert 'for condition in ("baseline", "+".join(names))' in text
    assert '"russian_language"' in text
    assert '"optimism"' in text
    assert '"atomic_sentences"' in text
    assert "default=3" in text
