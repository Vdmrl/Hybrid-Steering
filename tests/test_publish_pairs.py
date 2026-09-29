import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "publish_pairs",
    Path(__file__).parents[1] / "experiments" / "publish_pairs.py",
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
has_only_language_letters = _module.has_only_language_letters
text_units = _module.text_units


def test_new_scripts_reject_letters_from_another_script() -> None:
    assert has_only_language_letters("各个国家都受到财政制约。", "zh")
    assert not has_only_language_letters("国家 UN", "zh")
    assert has_only_language_letters("الخسارة المالية تصل للملايين", "ar")
    assert not has_only_language_letters("الخسارة UN", "ar")
    assert has_only_language_letters("ईमान लानेवालो", "hi")
    assert not has_only_language_letters("ईमान hello", "hi")


def test_chinese_length_counts_han_characters() -> None:
    assert text_units("在六日内创造天地万物", "zh") == 10
    assert text_units("twenty one english words here", "en") == 5
