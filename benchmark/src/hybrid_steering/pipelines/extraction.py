"""Extract a pinned target-minus-neutral direction from paired model answers."""

import argparse
from pathlib import Path

from hybrid_steering.extract import extract_directions
from hybrid_steering.pipelines.common import configure_data_root


def run(prompts: Path, config: Path, output: Path, data_root: Path) -> Path:
    root = configure_data_root(data_root)
    if not output.resolve().is_relative_to(root):
        raise ValueError("Extraction output must be inside --data-root")
    return extract_directions(prompts, config, output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="Folder with train.jsonl")
    parser.add_argument("--config", type=Path, required=True, help="Extraction JSON config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.prompts, args.config, args.output, args.data_root))


if __name__ == "__main__":
    main()
