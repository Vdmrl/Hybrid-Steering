import json

import pytest

from hybrid_steering import build_tiny, load_direction
from hybrid_steering.direction import (
    collect_from_pairs,
    concept_sides,
    read_pairs,
    target_and_source,
    write_direction,
)


def test_concept_sides_reads_a_slug_and_explicit_names() -> None:
    assert concept_sides("en-ru", None, None) == ("en", "ru")
    assert concept_sides("optimism", "plain", "hopeful") == ("plain", "hopeful")
    with pytest.raises(ValueError):
        concept_sides("optimism", None, None)


def test_target_text_is_the_positive_side(tmp_path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"positive_text": "target", "negative_text": "source"}) + "\n",
        encoding="utf-8",
    )
    assert target_and_source(read_pairs(path)[0]) == ("target", "source")


def test_collect_from_pairs_writes_a_loadable_direction(tmp_path) -> None:
    model, tokenizer = build_tiny()
    rows = [{"pair_id": "one", "positive_text": "aaaa", "negative_text": "bbbb"}]
    collected = collect_from_pairs(model, tokenizer, [target_and_source(rows[0])], batch_size=1)
    write_direction(tmp_path / "direction", "tiny", "ru", "en", rows, collected)
    direction, manifest, mean_target, mean_source = load_direction(tmp_path / "direction")
    assert manifest.target == "ru" and manifest.source == "en"
    assert manifest.example_ids == ["one"]
    assert set(direction) == set(mean_target) == set(mean_source)
