"""Collect language deltas, sweep ranks on SQuAD, and render one report."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the collection, steering sweep, and report stages."""
    parser = argparse.ArgumentParser(description="Run the language-steering pipeline.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language-a", default="en")
    parser.add_argument("--language-b", default="ru")
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--ranks", type=int, nargs="+", default=(2,))
    parser.add_argument("--scales", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--collection-batch-size", type=int, default=4)
    parser.add_argument("--prompt-steer-positions", type=int, nargs="+", default=(0, -1))
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--strict-script", action="store_true")
    parser.add_argument("--reject-digit-only-and-uppercase", action="store_true")
    parser.add_argument("--max-relative-token-difference", type=float)
    parser.add_argument("--prompt-steer-mode", choices=("exact", "chunk"), default="exact")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--cuda-devices", type=str, nargs="+")
    args = parser.parse_args()
    if min(args.pairs, args.questions, args.batch_size, args.collection_batch_size, args.min_words, args.max_new_tokens, args.parallelism) < 1:
        parser.error("counts and batch sizes must be positive")
    if args.language_a == args.language_b or any(rank < 0 for rank in args.ranks):
        parser.error("languages must differ and ranks must be non-negative")
    if args.max_relative_token_difference is not None and args.max_relative_token_difference < 0:
        parser.error("--max-relative-token-difference must be non-negative")
    if args.cuda_devices is not None and not args.cuda_devices:
        parser.error("--cuda-devices must name at least one GPU")

    args.output.mkdir(parents=True, exist_ok=True)
    deltas = args.output / "deltas.pt"
    sweep = args.output / "sweep"
    collect_command = [
        sys.executable,
        "-m",
        "experiments.steering.collect_acts",
        "--output",
        str(deltas),
        "--language-a",
        args.language_a,
        "--language-b",
        args.language_b,
        "--pairs",
        str(args.pairs),
        "--batch-size",
        str(args.collection_batch_size),
        "--min-words",
        str(args.min_words),
    ]
    if args.strict_script:
        collect_command.append("--strict-script")
    if args.reject_digit_only_and_uppercase:
        collect_command.append("--reject-digit-only-and-uppercase")
    if args.max_relative_token_difference is not None:
        collect_command.extend(("--max-relative-token-difference", str(args.max_relative_token_difference)))
    subprocess.run(collect_command, check=True)

    jobs = [(position, rank) for position in args.prompt_steer_positions for rank in dict.fromkeys(args.ranks)]
    processes: list[subprocess.Popen[bytes]] = []
    for job_index, (position, rank) in enumerate(jobs):
        output = sweep / f"position_{position}" / f"rank_{rank}"
        sweep_command = [
            sys.executable,
            "-m",
            "experiments.steering.squad_sweep",
            "--deltas",
            str(deltas),
            "--output",
            str(output),
            "--questions",
            str(args.questions),
            "--ranks",
            str(rank),
            "--scales",
            *(str(scale) for scale in args.scales),
            "--batch-size",
            str(args.batch_size),
            "--prompt-steer-position",
            str(position),
            "--prompt-steer-mode",
            str(args.prompt_steer_mode),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        sweep_command.append("--normalize" if args.normalize else "--no-normalize")
        environment = os.environ.copy()
        if args.cuda_devices is not None:
            environment["CUDA_VISIBLE_DEVICES"] = args.cuda_devices[job_index % len(args.cuda_devices)]
        processes.append(subprocess.Popen(sweep_command, env=environment))
        if len(processes) == args.parallelism:
            if any(process.wait() != 0 for process in processes):
                raise subprocess.CalledProcessError(1, sweep_command)
            processes.clear()
    if any(process.wait() != 0 for process in processes):
        raise subprocess.CalledProcessError(1, sweep_command)

    subprocess.run(
        [sys.executable, "-m", "experiments.steering.reports.report_ablations", str(sweep), "--output", str(args.output / "report.html")],
        check=True,
    )


if __name__ == "__main__":
    main()
