"""Generate saved benchmark answers, optionally score, or score saved HumanEval."""

import argparse
from pathlib import Path

from hybrid_steering.benchmarks import read_plan, run_benchmark
from hybrid_steering.pipelines.common import configure_data_root


def run(plan: Path, directions: Path, data_root: Path, *, score: bool) -> None:
    root = configure_data_root(data_root)
    benchmark = read_plan(plan)["benchmark"]
    if score and benchmark == "judge_prompts":
        raise ValueError("Judge prompts generate answers only; use gdn-concept to evaluate them")
    run_benchmark(plan, directions, root, score=score)


def score_humaneval(plan: Path, source_manifest: Path, data_root: Path) -> None:
    from hybrid_steering.score import score_saved

    root = configure_data_root(data_root)
    score_saved(plan, source_manifest, root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("generate", "run"):
        command = modes.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--directions", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
    score = modes.add_parser("score-humaneval", help="Score complete saved HumanEval answers offline")
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--source-manifest", type=Path, required=True)
    score.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "score-humaneval":
        score_humaneval(args.plan, args.source_manifest, args.data_root)
    else:
        run(args.plan, args.directions, args.data_root, score=args.mode == "run")


if __name__ == "__main__":
    main()
