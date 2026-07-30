"""Compose existing principled-candor and French GDN directions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]


def load_helpers() -> Any:
    path = ROOT / "experiments" / "calm-french-composition" / "run.py"
    spec = importlib.util.spec_from_file_location("calm_french_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = load_helpers()
COMPARISONS = {
    "candor_single": ("baseline", "candor"),
    "candor_with_french": ("french", "candor_french"),
    "french_single": ("baseline", "french"),
    "french_with_candor": ("candor", "candor_french"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--french-output", type=Path, required=True)
    parser.add_argument("--candor-direction", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--candor-alpha", type=float, default=8.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def write_judge_inputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    for name, (control, target) in COMPARISONS.items():
        path = output_dir / "judge-inputs" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "prompt_id": row["prompt_id"],
                        "scenario": row["scenario"],
                        "answers": [
                            {"answer_id": "control", "text": row[control]},
                            {"answer_id": "target", "text": row[target]},
                        ],
                        "metadata": {"comparison": name},
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for row in records
            ),
            encoding="utf-8",
        )


def main() -> None:
    args = arguments()
    if args.self_test:
        assert len(COMPARISONS) == 4
        assert COMPARISONS["candor_with_french"] == ("french", "candor_french")
        print("candor/French self-test passed")
        return

    prompts = HELPERS.jsonl(args.prompts)[: args.validation]
    if len(prompts) != args.validation:
        raise RuntimeError("incomplete validation prompt split")

    french = HELPERS.load_direction(
        args.french_output / "directions" / "french.safetensors"
    )
    candor = HELPERS.load_direction(args.candor_direction)
    french_alpha = float(
        json.loads(
            (args.french_output / "selection.json").read_text(encoding="utf-8")
        )["french_alpha"]
    )

    tokenizer, model = HELPERS.load_model(args.model)
    conditions = {
        "baseline": [],
        "candor": [(candor, args.candor_alpha)],
        "french": [(french, french_alpha)],
        "candor_french": [
            (candor, args.candor_alpha),
            (french, french_alpha),
        ],
    }
    generations = HELPERS.run_records(
        model=model,
        tokenizer=tokenizer,
        rows=prompts,
        conditions=conditions,
        path=args.output_dir / "generations.jsonl",
        max_new_tokens=args.max_new_tokens,
    )
    write_judge_inputs(generations, args.output_dir)
    HELPERS.atomic_json(
        args.output_dir / "language_summary.json",
        {
            condition: {
                "n": len(generations),
                "french_rate": sum(
                    HELPERS.is_french(row[condition]) for row in generations
                )
                / len(generations),
                "mean_words": sum(
                    len(row[condition].split()) for row in generations
                )
                / len(generations),
            }
            for condition in conditions
        },
    )
    HELPERS.atomic_json(
        args.output_dir / "run_manifest.json",
        {
            "model": args.model,
            "validation_prompts": args.validation,
            "conditions": list(conditions),
            "candor_alpha": args.candor_alpha,
            "french_alpha": french_alpha,
            "max_new_tokens": args.max_new_tokens,
            "judge_version": "v2",
            "reused_french_direction": str(args.french_output),
            "reused_candor_direction": str(args.candor_direction),
            "candor_french_cosine": HELPERS.flattened_cosine(candor, french),
        },
    )
    print("candor/French generation complete", flush=True)


if __name__ == "__main__":
    main()
