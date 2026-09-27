from metrics import metrics


def row(expected: int, predicted: int, feature: str = "optimism") -> dict:
    probabilities = {score: 0.02 for score in range(1, 6)}
    probabilities[predicted] = 0.92
    return {
        "expected_score": expected,
        "trait_score": predicted,
        "feature": feature,
        "score_distribution": {
            "probabilities": probabilities,
            "expected_score": float(predicted),
            "chosen_score_probability": 0.92,
            "entropy": 0.4,
        },
    }


def test_metrics_include_confusion_and_per_score() -> None:
    result = metrics([row(1, 1), row(3, 4), row(5, 5)])
    assert result["n"] == 3
    assert result["confusion"][0][0] == 1
    assert result["per_score"]["5"]["recall"] == 1
    assert result["high_confidence_errors"] == 1
    assert "by_feature" in result
