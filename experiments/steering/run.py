"""Score generations with a saved target-minus-source direction.

Build the direction first with ``hybrid-direction``. ``--feature`` selects the
score. A language id uses Lingua. Any other id uses the steering judge. The
default is the direction's target name, which for ``en-ru`` is ``ru``.
``--intervention clamp`` rewrites the rank-1 coordinate on every token; scale 0
is the unclamped baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_steering import Runner, load_direction, load_runtime
from hybrid_steering.detect import EVAL_QUESTIONS
from hybrid_steering.judge import score_rows
from hybrid_steering.runtime import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--feature", help="defaults to the direction target")
    parser.add_argument("--questions", type=int, default=0, help="0 uses every evaluation prompt")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intervention", choices=("add", "clamp"), default="add")
    args = parser.parse_args()
    direction, manifest, _, _ = load_direction(args.direction)
    questions = list(EVAL_QUESTIONS[: args.questions or None])
    model, tokenizer = load_runtime(args.model)
    runner = Runner.from_direction(
        model,
        tokenizer,
        direction,
        rank=args.rank or None,
        normalize=args.normalize,
        intervention=args.intervention,
    )
    feature = args.feature or manifest.target
    rows = []
    for scale in args.scales:
        tokens = runner.generate(
            questions,
            scale=scale,
            prompt_position=args.position,
            max_new_tokens=args.max_new_tokens,
        )
        for prompt, row_tokens in zip(questions, tokens, strict=True):
            text = tokenizer.decode(row_tokens, skip_special_tokens=True)
            rows.append(
                {
                    "scale": scale,
                    "prompt": prompt,
                    "response": text,
                    "target": manifest.target,
                    "source": manifest.source,
                }
            )
    score_rows(rows, feature)
    write_jsonl(args.output / "generations.jsonl", rows)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
