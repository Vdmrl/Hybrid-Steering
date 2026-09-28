from hybrid_steering import concept_detector


def test_language_feature_uses_lingua() -> None:
    detector = concept_detector("russian_language")
    assert detector.detects("Это длинное русское предложение про дождь, город и море.")
    assert not detector.detects("This is a long English sentence about rain, a city, and the sea.")
    assert concept_detector("ru").target == "ru"


def test_other_concepts_use_the_supplied_verdict() -> None:
    detector = concept_detector("optimism", verdict="0")
    assert detector.target
    assert not detector.detects("anything", question="What next?")
    assert detector.label("anything", question="What next?") == "0"
