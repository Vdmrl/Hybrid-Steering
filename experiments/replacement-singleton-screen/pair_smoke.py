"""Generate a minimal confident + optimism RSS/clamp smoke."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
FINAL_DIR = ROOT / "experiments" / "final-feature-screen"
sys.path.insert(0, str(FINAL_DIR))
spec = importlib.util.spec_from_file_location("final_run", FINAL_DIR / "final_run.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load final_run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--confident", type=Path, required=True)
    parser.add_argument("--optimism", type=Path, required=True)
    parser.add_argument("--confident-alpha", type=float, default=4)
    parser.add_argument("--optimism-alpha", type=float, default=4)
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [
        {
            "name": "confident",
            "direction": str(args.confident),
            "rank": "rank1",
            "alpha": args.confident_alpha,
            "c": 1,
        },
        {
            "name": "optimism",
            "direction": str(args.optimism),
            "rank": "rank1",
            "alpha": args.optimism_alpha,
            "c": 1,
        },
    ]
    directions = {
        item["name"]: run.tensor_map(Path(item["direction"]), 1) for item in selected
    }
    tokenizer, model = run.runner.load_model("Qwen/Qwen3.5-9B")
    output = args.output / "pair-generations.jsonl"
    for row in run.prompt_rows(args.prompts, args.limit):
        base = run.runner.prefill(model, tokenizer, row["prompt"])
        for subset in ((), ("confident",), ("optimism",), ("confident", "optimism")):
            chosen = [item for item in selected if item["name"] in subset]
            if chosen:
                response, _ = run.generate_condition(
                    model, tokenizer, base, directions, chosen, 128, 1
                )
            else:
                response = run.runner.decode(
                    model, tokenizer, run.runner.clone_cache(base), 128
                )
            run.append_jsonl(
                output,
                {
                    "prompt_id": row["id"],
                    "scenario": row["prompt"],
                    "condition": "+".join(subset) or "baseline",
                    "active_features": list(subset),
                    "response": response,
                },
            )
            print(row["id"], "+".join(subset) or "baseline", flush=True)


if __name__ == "__main__":
    main()
