"""Extract one rank-4 direction from prepared JSONL pairs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from safetensors.torch import save_file

ROOT = Path(__file__).parents[2]
BASE = ROOT / "experiments" / "style-singleton-screen" / "run.py"
spec = importlib.util.spec_from_file_location("style_screen_runner", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE}")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer, model = runner.load_model("Qwen/Qwen3.5-9B")
    diffs = []
    for index, pair in enumerate(pairs, 1):
        positive = runner.extract_recurrent(
            runner.prefill(model, tokenizer, pair["positive_text"])
        )
        negative = runner.extract_recurrent(
            runner.prefill(model, tokenizer, pair["negative_text"])
        )
        diffs.append(runner.subtract_states(positive, negative))
        if index % 16 == 0:
            print(f"direction: {index}/{len(pairs)}", flush=True)
    direction = runner.low_rank(runner.mean_direction(diffs), 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {f"layer_{layer}": value.contiguous() for layer, value in direction.items()},
        str(args.output),
    )


if __name__ == "__main__":
    main()
