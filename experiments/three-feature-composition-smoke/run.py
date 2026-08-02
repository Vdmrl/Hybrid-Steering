"""Generate only baseline and the Russian+optimism+atomic composition."""

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
    parser.add_argument("--russian", type=Path, required=True)
    parser.add_argument("--optimism", type=Path, required=True)
    parser.add_argument("--atomic", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--atomic-alpha", type=float, default=3)
    parser.add_argument("--scale", type=float, default=0.85)
    parser.add_argument("--russian-scale", type=float, default=1)
    parser.add_argument("--tag", default="atomic3")
    return parser.parse_args()


def append(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    args = arguments()
    features = [
        {
            "name": "russian_language",
            "direction": str(args.russian),
            "rank": "rank4",
            "alpha": 1.5,
            "c": args.russian_scale,
        },
        {
            "name": "optimism",
            "direction": str(args.optimism),
            "rank": "rank1",
            "alpha": 4,
            "c": args.scale,
        },
        {
            "name": "atomic_sentences",
            "direction": str(args.atomic),
            "rank": "rank4",
            "alpha": args.atomic_alpha,
            "c": args.scale,
        },
    ]
    directions = {
        item["name"]: run.tensor_map(Path(item["direction"]), 1) for item in features
    }
    names = tuple(item["name"] for item in features)
    output = args.output / f"{args.tag}.jsonl"
    done = (
        {row["task_id"] for row in run.read_jsonl(output)} if output.exists() else set()
    )
    tokenizer, model = run.runner.load_model("Qwen/Qwen3.5-9B")
    for row in run.prompt_rows(args.prompts, args.limit):
        base = run.runner.prefill(model, tokenizer, row["prompt"])
        for condition in ("baseline", "+".join(names)):
            task_id = f"{row['id']}:{condition}"
            if task_id in done:
                continue
            if condition == "baseline":
                response = run.clean_response(
                    run.runner.decode(
                        model, tokenizer, run.runner.clone_cache(base), 128
                    )
                )
            else:
                response, trace = run.generate_condition(
                    model, tokenizer, base, directions, features, 128, 1
                )
            append(
                output,
                {
                    "task_id": task_id,
                    "prompt_id": row["id"],
                    "scenario": row["prompt"],
                    "condition": condition,
                    "active_features": [] if condition == "baseline" else list(names),
                    "response": response,
                    "selection": features,
                    "coefficient_trace": [] if condition == "baseline" else trace,
                },
            )
            print(task_id, flush=True)


if __name__ == "__main__":
    main()
