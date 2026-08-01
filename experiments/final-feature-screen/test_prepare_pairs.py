from prepare_pairs import headline_from_edit, numbered_items, validate


def test_headline_edit_and_numbered_validation() -> None:
    assert (
        headline_from_edit("Dog <runs/> across field", "dances")
        == "Dog dances across field"
    )
    assert numbered_items("- First item\n- Second item\n- Third item") == [
        "First item",
        "Second item",
        "Third item",
    ]
    rows = [
        {
            "source_id": "x",
            "negative_text": "one two three four five six seven eight nine",
            "positive_text": "1. one two three\n2. four five six\n3. seven eight nine",
        }
    ]
    validate(rows, numbered=True)
