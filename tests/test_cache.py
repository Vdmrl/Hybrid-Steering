from pathlib import Path
from types import SimpleNamespace

import torch

from hybrid_steering import (
    DirectionManifest,
    apply_direction,
    assert_nonrecurrent_unchanged,
    clone_cache,
    difference,
    extract_recurrent,
    gdn_layer_indices,
    load_direction,
    mean,
    save_direction,
    snapshot_nonrecurrent,
)


def fake_cache():
    gdn = SimpleNamespace(
        recurrent_states={0: torch.zeros(1, 2, 4, 4)},
        conv_states={0: torch.ones(1, 6, 4)},
    )
    attention = SimpleNamespace(
        keys=[torch.ones(1, 2, 3, 4)],
        values=[torch.full((1, 2, 3, 4), 2.0)],
    )
    return SimpleNamespace(layers=[gdn, attention])


def test_apply_direction_preserves_nonrecurrent_cache(tmp_path: Path) -> None:
    cache = clone_cache(fake_cache())
    assert gdn_layer_indices(cache) == [0]
    before = snapshot_nonrecurrent(cache)
    target = {0: torch.ones(1, 2, 4, 4)}
    source = extract_recurrent(cache)
    direction = mean([difference(target, source)])
    apply_direction(cache, direction, 0.5, layers=[0])
    torch.testing.assert_close(extract_recurrent(cache)[0], torch.full((1, 2, 4, 4), 0.5))
    assert_nonrecurrent_unchanged(before, cache)

    manifest = DirectionManifest(
        model_id="tiny",
        target="russian",
        source="english",
        example_ids=["pair-001"],
        decoder_layer_indices=[0],
        state_shapes={0: [1, 2, 4, 4]},
    )
    save_direction(tmp_path, direction, manifest)
    loaded, loaded_manifest, mean_target, mean_source = load_direction(tmp_path)
    torch.testing.assert_close(loaded[0], direction[0])
    assert loaded_manifest == manifest
    assert mean_target is None and mean_source is None
