from pole_metrics import auc, feature_metrics, join_gold, pole_metrics


def row(prompt_id, pole, trait, expected, pair_id=None, feature="optimism"):
    return {
        "prompt_id": prompt_id,
        "feature": feature,
        "pole": pole,
        "pair_id": pair_id,
        "trait_score": float(trait),
        "expected_score": float(expected),
    }


def test_auc_is_half_with_identical_distributions():
    assert auc([3.0, 4.0], [3.0, 4.0]) == 0.5


def test_auc_counts_ties_as_half_and_is_one_when_separated():
    assert auc([5.0], [3.0]) == 1.0
    assert auc([4.0, 4.0], [4.0, 2.0]) == 0.75


def test_auc_is_nan_when_one_side_is_empty():
    value = auc([4.0], [])
    assert value != value


def test_feature_metrics_reports_gap_effect_size_and_saturation():
    rows = [
        row("a", "target", 5, 4.8, pair_id="p1"),
        row("b", "opposite", 2, 2.1, pair_id="p1"),
        row("c", "target", 4, 3.9, pair_id="p2"),
        row("d", "opposite", 1, 1.4, pair_id="p2"),
    ]
    result = feature_metrics(rows)
    assert result["auc"] == 1.0
    assert result["cliffs_delta"] == 1.0
    assert result["mean_gap"] == 3.0
    assert result["saturation_high"] == 0.25
    assert result["paired_trait_score"]["pairs"] == 2
    assert result["paired_trait_score"]["pair_win_rate"] == 1.0
    assert result["paired_trait_score"]["pair_tie_rate"] == 0.0


def test_paired_section_ignores_unpaired_rows():
    rows = [row("a", "target", 5, 4.8), row("b", "opposite", 2, 2.1)]
    assert feature_metrics(rows)["paired_trait_score"] == {"pairs": 0}


def test_join_gold_matches_by_prompt_id_and_drops_unlabeled():
    results = [
        {
            "prompt_id": "kept",
            "feature": "optimism",
            "trait_score": 4,
            "score_distribution": {"expected_score": 3.7},
        },
        {
            "prompt_id": "unlabeled",
            "feature": "optimism",
            "trait_score": 2,
            "score_distribution": {"expected_score": 2.2},
        },
    ]
    gold = {"kept": {"prompt_id": "kept", "pole": "target", "concept": "optimism"}}
    rows = join_gold(results, gold)
    assert [r["prompt_id"] for r in rows] == ["kept"]
    assert rows[0]["pole"] == "target"
    assert rows[0]["expected_score"] == 3.7


def test_pole_metrics_aggregates_per_feature():
    rows = [
        row("a", "target", 5, 4.8, feature="optimism"),
        row("b", "opposite", 1, 1.2, feature="optimism"),
        row("c", "target", 3, 3.0, feature="casualness"),
        row("d", "opposite", 3, 3.0, feature="casualness"),
    ]
    result = pole_metrics(rows)
    assert result["features"] == 2
    assert result["by_feature"]["optimism"]["auc"] == 1.0
    assert result["by_feature"]["casualness"]["auc"] == 0.5
    assert result["macro_auc"] == 0.75
    assert result["min_auc"] == 0.5
