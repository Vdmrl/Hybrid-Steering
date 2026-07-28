from pathlib import Path
from types import SimpleNamespace

import torch

from hybrid_steering import (
    DirectionManifest,
    add_direction,
    assert_nonrecurrent_unchanged,
    clone_cache,
    extract_recurrent,
    gdn_layer_indices,
    load_direction,
    mean_direction,
    save_direction,
    snapshot_nonrecurrent,
    subtract_states,
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


def test_zero_based_intervention_preserves_other_cache(tmp_path: Path) -> None:
    cache = clone_cache(fake_cache())
    assert gdn_layer_indices(cache) == [0]
    before = snapshot_nonrecurrent(cache)

    positive = {0: torch.ones(1, 2, 4, 4)}
    negative = extract_recurrent(cache)
    direction = mean_direction([subtract_states(positive, negative)])
    add_direction(cache, direction, 0.5, layers=[0])

    torch.testing.assert_close(
        extract_recurrent(cache)[0],
        torch.full((1, 2, 4, 4), 0.5),
    )
    assert_nonrecurrent_unchanged(before, cache)

    manifest = DirectionManifest(
        model_id="Qwen/Qwen3.5-9B",
        transformers_version="5.14.1",
        positive_pole="optimism",
        negative_pole="pessimism",
        train_example_ids=["pair-001"],
        decoder_layer_indices_zero_based=[0],
        state_shapes={0: [1, 2, 4, 4]},
    )
    save_direction(tmp_path, direction, manifest)
    loaded, loaded_manifest = load_direction(tmp_path)
    torch.testing.assert_close(loaded[0], direction[0])
    assert loaded_manifest == manifest
