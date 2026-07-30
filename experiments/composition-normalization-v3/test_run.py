import importlib.util
from pathlib import Path

MODULE = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("composition_normalization_v3", MODULE)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_composition_helpers():
    RUN.self_test()


def test_bootstrap_mean_is_deterministic():
    first = RUN.bootstrap_mean([1.0, 2.0, 3.0], samples=100)
    second = RUN.bootstrap_mean([1.0, 2.0, 3.0], samples=100)
    assert first == second
    assert first["mean"] == 2.0
