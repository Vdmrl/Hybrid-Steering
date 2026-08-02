"""Smoke-test atomic sentences with one existing feature."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
FINAL = ROOT / "experiments" / "final-feature-screen" / "final_run.py"
sys.path.insert(0, str(FINAL.parent))
spec = importlib.util.spec_from_file_location("final_run", FINAL)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {FINAL}")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--atomic", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument(
        "--other-name", choices=("optimism", "numbered_list"), required=True
    )
    parser.add_argument("--other-rank", choices=("rank1", "rank4"), required=True)
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def append(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    args = arguments()
    features = [
        {
            "name": "atomic_sentences",
            "direction": str(args.atomic),
            "rank": "rank4",
            "alpha": 4,
            "c": 1,
        },
        {
            "name": args.other_name,
            "direction": str(args.other),
            "rank": args.other_rank,
            "alpha": 4,
            "c": 1,
        },
    ]
    directions = {
        item["name"]: run.tensor_map(Path(item["direction"]), 1) for item in features
    }
    output = args.output / f"atomic+{args.other_name}.jsonl"
    done = (
        {row["task_id"] for row in run.read_jsonl(output)} if output.exists() else set()
    )
    tokenizer, model = run.runner.load_model("Qwen/Qwen3.5-9B")
    names = ("atomic_sentences", args.other_name)
    for row in run.prompt_rows(args.prompts, args.limit):
        base = run.runner.prefill(model, tokenizer, row["prompt"])
        for subset in ((), names[:1], names[1:], names):
            condition = "+".join(subset) or "baseline"
            task_id = f"{row['id']}:{condition}"
            if task_id in done:
                continue
            if subset:
                chosen = [item for item in features if item["name"] in subset]
                response, _ = run.generate_condition(
                    model, tokenizer, base, directions, chosen, 128, 1
                )
            else:
                response = run.runner.decode(
                    model, tokenizer, run.runner.clone_cache(base), 128
                )
            append(
                output,
                {
                    "task_id": task_id,
                    "prompt_id": row["id"],
                    "scenario": row["prompt"],
                    "condition": condition,
                    "active_features": list(subset),
                    "response": response,
                    "selection": features,
                },
            )
            print(task_id, flush=True)


if __name__ == "__main__":
    main()
