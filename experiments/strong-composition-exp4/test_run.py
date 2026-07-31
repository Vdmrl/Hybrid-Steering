import importlib.util
from pathlib import Path

MODULE = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("strong_composition_exp4", MODULE)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_experiment_contract():
    RUN.self_test()
