"""CPU contracts; no model loading, generation or Judge calls."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

# Load the real cache/state modules without the package's model/provider imports.
ROOT = Path(__file__).resolve().parents[2]
original_modules = dict(sys.modules)
package = ModuleType("hybrid_steering")
package.__path__ = [str(ROOT / "src/hybrid_steering")]
sys.modules.setdefault("hybrid_steering", package)
SPEC = importlib.util.spec_from_file_location(
    "gdn_clamp_runtime", Path(__file__).with_name("runtime.py")
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)
for name in list(sys.modules):
    if name.startswith("hybrid_steering") and name not in original_modules:
        del sys.modules[name]


def cache():
    return SimpleNamespace(
        layers=[
            SimpleNamespace(
                recurrent_states=torch.zeros(1, 1, 1, 3),
                keys=torch.ones(1, 2),
                conv_states=torch.ones(1, 3),
            )
        ]
    )


def test_separate_betas_and_zero_replay():
    c = cache()
    directions = {
        "a": {0: torch.tensor([[[[1.0, 0.0, 0.0]]]])},
        "b": {0: torch.tensor([[[[0.0, 1.0, 0.0]]]])},
    }
    before = runtime.snapshot_nonrecurrent(c)
    state = runtime.make_runtime(c, directions, ["a", "b"], {"a": 2.0, "b": 4.0})
    runtime.apply_clamp(c, state, {"a": 0.0, "b": 0.0})
    assert torch.equal(c.layers[0].recurrent_states, torch.zeros(1, 1, 1, 3))
    runtime.apply_clamp(c, state, {"a": 0.25, "b": 0.5})
    torch.testing.assert_close(
        c.layers[0].recurrent_states.flatten(), torch.tensor([0.5, 2.0, 0.0])
    )
    runtime.apply_clamp(c, state, {"a": 0.25, "b": 0.5})
    torch.testing.assert_close(
        c.layers[0].recurrent_states.flatten(), torch.tensor([0.875, 3.0, 0.0])
    )
    runtime.assert_nonrecurrent_unchanged(before, c)
    replay = cache()
    other = runtime.make_runtime(replay, directions, ["a", "b"], {"a": 2.0, "b": 4.0})
    for _ in range(2):
        runtime.apply_clamp(replay, other, {"a": 0.25, "b": 0.5})
    torch.testing.assert_close(replay.layers[0].recurrent_states, c.layers[0].recurrent_states)


def test_initial_full_clamp_and_rank1_singleton_rss():
    c = cache()
    directions = {"a": {0: torch.tensor([[[[1.0, 0.0, 0.0]]]])}}
    assert runtime.rss_coefficients(directions, {"a": 2.0}) == {0: {"a": 2.0}}
    state = runtime.make_runtime(c, directions, ["a"], {"a": 2.0})
    runtime.apply_clamp(c, state, {"a": 1.0})
    torch.testing.assert_close(
        c.layers[0].recurrent_states.flatten(), torch.tensor([2.0, 0.0, 0.0])
    )
    runtime.finite_active(c, state)


def test_rss_composition_and_layer_routing():
    d = torch.tensor([1.0, 0.0])
    coefficients = runtime.rss_coefficients({"a": {0: d}, "b": {0: d, 1: d}}, {"a": 1.0, "b": 1.0})
    assert abs(coefficients[0]["a"] - 2**-0.5) < 1e-6
    assert coefficients[1] == {"b": 1.0}
