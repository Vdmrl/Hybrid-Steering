"""Offline contract checks for the separately installed benchmark handoff."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmark"


def test_russian_rate_implementation_is_frozen() -> None:
    source = BUNDLE / "src/hybrid_steering/language_rate.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "11acba139582672a2c891eb22866b0650a5ce30ea8d20792df90dc4be97a97be"
    )


def test_offline_plan_covers_all_models_languages_and_requested_counts(tmp_path: Path) -> None:
    concepts = tmp_path / "concepts"
    for slug in ("en-ru", "en-fr", "en-ar"):
        path = concepts / "concepts" / slug / "data/pairs.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8") as stream:
            for index in range(2000):
                stream.write(json.dumps({
                    "pair_id": f"{slug}-{index}",
                    "split": "train",
                    "negative_text": f"English prompt {index}",
                    "positive_text": f"Target answer {index}",
                }) + "\n")
    environment = {**os.environ, "PYTHONPATH": f"{BUNDLE / 'src'}:{BUNDLE}"}
    result = subprocess.run(
        [sys.executable, "-m", "experiments.language_pareto_handoff", "plan",
         "--concept-root", str(concepts), "--data-root", str(tmp_path / "output")],
        cwd=BUNDLE,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    lines = result.stdout.splitlines()
    assert len(lines) == 6
    assert {line.split(":", 1)[0] for line in lines} == {
        f"{model} {language}"
        for model in ("qwen", "falcon")
        for language in ("russian", "french", "arabic")
    }
    assert all("97 screen conditions × 100 prompts" in line for line in lines)
    assert all("40 selected conditions × (500+541+164) prompts" in line for line in lines)


def test_residual_clamp_fixes_projection_and_skips_prefill() -> None:
    pytest.importorskip("torch")
    environment = {**os.environ, "PYTHONPATH": f"{BUNDLE / 'src'}:{BUNDLE}"}
    program = """
import torch
from hybrid_steering.residual import residual_hook

block = torch.nn.Identity()
unit = torch.tensor([1.0, 0.0])
prefill = torch.tensor([[[0.5, 3.0], [1.0, 4.0]]])
with residual_hook(block, unit, clamp_target=torch.tensor(2.0)):
    assert torch.equal(block(prefill.clone()), prefill)
    first = block(torch.tensor([[[0.5, 5.0]]]))
    assert torch.allclose(first, torch.tensor([[[2.0, 5.0]]]))
    assert torch.equal(block(first.clone()), first)
"""
    subprocess.run(
        [sys.executable, "-c", program], cwd=BUNDLE, env=environment,
        capture_output=True, text=True, timeout=30, check=True,
    )
