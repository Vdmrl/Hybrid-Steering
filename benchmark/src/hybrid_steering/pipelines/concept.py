"""Prepare blinded concept tasks or explicitly send paid Judge requests."""

import argparse

from hybrid_steering.judge.__main__ import main as judge_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "run"):
        command = modes.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--results", required=False)
        source.add_argument("--input", required=False)
        command.add_argument("--concept")
        command.add_argument("--output")
        command.add_argument("--settings")
        command.add_argument("--data-root")
        command.add_argument("--key-file")
        command.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    forwarded = []
    for name in ("results", "input", "concept", "output", "settings", "data_root",
                 "key_file", "workers"):
        value = getattr(args, name)
        if value is not None:
            forwarded.extend(("--" + name.replace("_", "-"), str(value)))
    if args.mode == "run":
        forwarded.append("--run")
    judge_main(forwarded)


if __name__ == "__main__":
    main()
