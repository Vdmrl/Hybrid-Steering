from pathlib import Path

SOURCE = Path(__file__).with_name("prepare.py")


def test_preflight_has_three_requested_features():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"russian_language": prepare_russian' in text
    assert '"optimism": prepare_optimism' in text
    assert '"atomic_sentences": (' in text
    assert "prepare_atomic_generated" in text
