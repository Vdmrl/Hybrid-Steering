import pytest

from hybrid_steering import concept_detector


def test_language_feature_uses_lingua() -> None:
    detector = concept_detector("russian_language")
    assert detector.detects("Это длинное русское предложение про дождь, город и море.")
    assert not detector.detects("This is a long English sentence about rain, a city, and the sea.")
    assert concept_detector("ru").target == "ru"


def test_added_languages_use_lingua_without_changing_russian() -> None:
    cases = {
        "chinese_language": "这是一个关于雨、城市和大海的很长的中文句子。",
        "arabic_language": "هذه جملة عربية طويلة عن المطر والمدينة والبحر.",
        "hindi_language": "यह बारिश, शहर और समुद्र के बारे में एक लंबा हिंदी वाक्य है।",
    }
    english = "This is a long English sentence about rain, a city, and the sea."
    for feature, text in cases.items():
        detector = concept_detector(feature)
        assert detector.detects(text)
        assert not detector.detects(english)
    # Russian keeps the candidate set its published scores used
    assert len(concept_detector("ru").languages) == 3


def test_non_language_features_are_not_a_detector() -> None:
    with pytest.raises(ValueError, match="optimism"):
        concept_detector("optimism")
