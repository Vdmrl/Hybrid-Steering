"""One CLI for extraction → steered model → benchmark; run from project root."""

import argparse
import json
from pathlib import Path

from hybrid_steering.benchmarks import read_plan, run_benchmark
from hybrid_steering.extract import extract_directions
from hybrid_steering.paths import project_file
from hybrid_steering.pipelines.benchmark import run as run_benchmark_pipeline
from hybrid_steering.pipelines.common import configure_data_root
from hybrid_steering.pipelines.extraction import run as run_extraction_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="Generate paired answers and directions")
    extract.add_argument("--prompts", type=Path, required=True, help="Folder with train.jsonl")
    extract.add_argument("--config", type=Path, required=True, help="Extraction JSON config")
    extract.add_argument("--output", type=Path, required=True, help="New direction output folder")
    benchmark = commands.add_parser("benchmark", help="Run a pinned benchmark or generate Judge-prompt answers")
    benchmark.add_argument("--directions", type=Path, required=True)
    benchmark.add_argument("--config", type=Path, required=True, help="Benchmark JSON plan")
    benchmark.add_argument("--generate-only", action="store_true")
    combined = commands.add_parser("all", help="Extract and benchmark with one model load")
    combined.add_argument("--prompts", type=Path, required=True)
    combined.add_argument("--extract-config", type=Path, required=True)
    combined.add_argument("--benchmark-config", type=Path, required=True)
    combined.add_argument("--output", type=Path, required=True)
    combined.add_argument("--generate-only", action="store_true")
    for command in (extract, benchmark, combined):
        command.add_argument("--data-root", type=Path, required=True,
                             help="Host-specific root for models, caches, directions and results")
    args = parser.parse_args()
    data_root = configure_data_root(args.data_root)

    if args.command == "extract":
        path = run_extraction_pipeline(args.prompts, args.config, args.output, data_root)
        print(path)
    elif args.command == "benchmark":
        run_benchmark_pipeline(args.config, args.directions, data_root,
                               score=not args.generate_only)
    else:
        if not args.output.resolve().is_relative_to(data_root):
            parser.error("Extraction output must be inside --data-root")
        extraction = json.loads(args.extract_config.read_text())
        plan = read_plan(args.benchmark_config)
        if extraction["model"] != plan["model"]:
            parser.error("Extraction and benchmark must use the same model")
        from hybrid_steering.language_benchmark import load_model
        placement = None
        if plan.get("placement_config"):
            placement = json.loads(project_file(plan["placement_config"]).read_text())["placement"]
        model, tokenizer = load_model(plan["model"], placement)
        path = extract_directions(args.prompts, args.extract_config, args.output, model, tokenizer)
        run_benchmark(args.benchmark_config, path, data_root,
                      score=not args.generate_only, model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
