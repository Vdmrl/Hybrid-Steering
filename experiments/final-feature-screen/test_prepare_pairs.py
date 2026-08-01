from prepare_pairs import headline_from_edit, numbered_items, prepare_numbered, validate


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


def test_numbered_source_accepts_jsonl(tmp_path) -> None:
    source = tmp_path / "bullets.jsonl"
    source.write_text(
        '{"source_id":"x","positive_text":"- First item\\n- Second item\\n- Third item","negative_text":"First item. Second item. Third item."}\n',
        encoding="utf-8",
    )
    pairs = prepare_numbered(source, 1)
    assert pairs[0]["positive_text"].startswith("1. First item")
