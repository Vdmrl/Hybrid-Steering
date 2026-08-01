from build_compositional_benchmark import SCENARIOS, extra_labels, labels


def test_v3_scenario_groups_do_not_cross_splits() -> None:
    by_split = {
        split: {
            scenario
            for scenario, (_, row_split) in SCENARIOS.items()
            if row_split == split
        }
        for split in ("development", "validation")
    }
    assert by_split["development"]
    assert by_split["validation"]
    assert by_split["development"].isdisjoint(by_split["validation"])


def test_v3_labels_observable_traits() -> None:
    assert labels("presentation", "singleton_optimism")["optimism"] == 4
    assert labels("zoning", "singleton_optimism")["optimism"] == 3
    assert labels("campaign", "flip_bulleted_layout")["optimism"] == 3
    assert labels("allocation", "flip_optimism")["optimism"] == 1
    assert labels("zoning", "flip_french_language")["bulleted_layout"] == 5
    assert extra_labels("bullets_negative")["first_person_voice"] == 5
