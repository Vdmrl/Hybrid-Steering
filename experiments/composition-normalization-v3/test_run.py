import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_quality_coverage_accepts_ninety_percent(tmp_path):
    root = tmp_path / "judge" / "dev"
    inputs = [
        {
            "prompt_id": str(index),
            "scenario": "test",
            "answers": [{"answer_id": "answer", "text": "text"}],
        }
        for index in range(10)
    ]
    results = [{"prompt_id": str(index), "answer_id": "answer"} for index in range(9)]
    RUN.write_jsonl(root / "inputs" / "quality.jsonl", inputs)
    RUN.write_jsonl(root / "results" / "quality.jsonl", results)
    RUN.check_quality_coverage(SimpleNamespace(output_dir=tmp_path, split="dev"))
    report = json.loads((root / "quality-coverage.json").read_text())
    assert report["overall"] == 0.9
