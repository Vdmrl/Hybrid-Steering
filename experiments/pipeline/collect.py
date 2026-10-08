"""Score saved language sweeps and write SVG plots plus JSON traces.

Language concept scores come from Lingua. IFEval and HumanEval use the official
scorers. The script reads answer directories that are already complete, so a
later edit to the steering code does not hide those generations.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import _instruction_interval, _interval, _sandbox_command, _sha
from figures import write_svgs
from report import build_report

from hybrid_steering.detect import concept_detector
from hybrid_steering.runtime import write_jsonl
from hybrid_steering.scoring import choose, summarize

FEATURES = {
    "russian": "russian_language",
    "french": "french_language",
    "chinese": "chinese_language",
    "arabic": "arabic_language",
}


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def expected_rows(manifest: dict) -> int:
    return manifest["n_tasks"] * sum(len(case["scales"]) for case in manifest["conditions"])


def code_key(manifest: dict) -> str:
    return manifest["code_sha256"]["experiments/pipeline/residual.py"]


def parse_run(name: str) -> tuple[str, str, str]:
    prefix, language, *rest = name.split("-")
    if language not in FEATURES or not rest:
        raise ValueError(f"unrecognised run name {name}")
    return prefix, language, "-".join(rest)


def locate(path: str, remap: list[tuple[str, str]]) -> Path:
    for source, target in remap:
        if path.startswith(source):
            return Path(target + path[len(source) :])
    return Path(path)


def answer_dirs(run: Path) -> list[tuple[Path, dict, list[dict]]]:
    found = []
    for manifest_path in sorted(run.glob("*/*/manifest.json")):
        directory = manifest_path.parent
        answers_path = directory / "answers.jsonl"
        if not answers_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        answers = read_jsonl(answers_path)
        if len(answers) != expected_rows(manifest):
            continue
        found.append((directory, manifest, answers))
    return found


def score_language(directory: Path, manifest: dict, answers: list[dict], feature: str) -> None:
    scores = directory / "judge_scores.jsonl"
    if scores.exists() and sum(1 for line in scores.open() if line.strip()) == len(answers):
        return
    detector = concept_detector(feature)
    rated = []
    for row in answers:
        concept = int(detector.detects(row["response"]))
        rated.append({**row, "question": row["prompt"], "concept_score": concept, "hit": concept})
    write_jsonl(scores, rated)
    plan = {"conditions": manifest["conditions"]}
    cells = summarize(rated)
    methods = {
        case["name"]: case.get("rank")
        for case in plan["conditions"]
        if case["method"] != "baseline"
    }
    chosen = choose(cells, methods=methods) if methods else {}
    (directory / "sweep_summary.json").write_text(
        json.dumps({"cells": _clean(cells), "chosen": _clean(chosen)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"lingua {directory.parent.parent.name}", flush=True)


def _clean(value):
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def score_ifeval(
    directory: Path, manifest: dict, answers: list[dict], remap: list[tuple[str, str]]
) -> None:
    if (directory / "benchmark_scores.json").exists() and (
        directory / "item_scores.jsonl"
    ).exists():
        return
    spec = manifest["dataset"]
    dataset_path = locate(spec["path"], remap)
    root = locate(spec["evaluator"]["root"], remap)
    source = root / "instruction_following_eval/evaluation_lib.py"
    if _sha(source) != spec["evaluator"]["evaluation_lib_sha256"]:
        raise ValueError(f"IFEval evaluator hash differs for {directory}")
    sys.path.insert(0, str(root))
    try:
        from instruction_following_eval import evaluation_lib
    finally:
        sys.path.pop(0)
    inputs = evaluation_lib.read_prompt_list(dataset_path)
    items = []
    metrics = {}
    for case in manifest["conditions"]:
        for scale in case["scales"]:
            selected = [
                row
                for row in answers
                if row["condition"] == case["name"] and row["scale"] == float(scale)
            ]
            response = {row["prompt"]: row["response"] for row in selected}
            keys = {row["prompt"]: row["key"] for row in selected}
            random.seed(0)
            strict = [
                evaluation_lib.test_instruction_following_strict(row, response) for row in inputs
            ]
            random.seed(0)
            loose = [
                evaluation_lib.test_instruction_following_loose(row, response) for row in inputs
            ]
            for prompt_row, strict_row, loose_row in zip(inputs, strict, loose, strict=True):
                items.append(
                    {
                        "key": keys[prompt_row.prompt],
                        "condition": case["name"],
                        "scale": float(scale),
                        "prompt_strict": bool(strict_row.follow_all_instructions),
                        "prompt_loose": bool(loose_row.follow_all_instructions),
                    }
                )
            cell = {
                "prompt_strict": sum(row.follow_all_instructions for row in strict) / len(strict),
                "prompt_loose": sum(row.follow_all_instructions for row in loose) / len(loose),
                "instruction_strict": sum(sum(row.follow_instruction_list) for row in strict)
                / sum(len(row.follow_instruction_list) for row in strict),
                "instruction_loose": sum(sum(row.follow_instruction_list) for row in loose)
                / sum(len(row.follow_instruction_list) for row in loose),
            }
            cell["confidence_intervals"] = {
                "prompt_strict": _interval([bool(row.follow_all_instructions) for row in strict]),
                "prompt_loose": _interval([bool(row.follow_all_instructions) for row in loose]),
                "instruction_strict": _instruction_interval(strict),
                "instruction_loose": _instruction_interval(loose),
            }
            metrics[f"{case['name']}:{scale}"] = cell
    write_jsonl(directory / "item_scores.jsonl", items)
    (directory / "benchmark_scores.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"ifeval {directory.parent.parent.name}", flush=True)


def _humaneval_one(payload: str) -> bool:
    import subprocess

    command = _sandbox_command(Path(__file__).with_name("humaneval_execute_one.py").resolve())
    try:
        done = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=10)
    except subprocess.TimeoutExpired:
        return False
    if done.returncode:
        return False
    return bool(json.loads(done.stdout)["passed"])


def score_humaneval(
    directory: Path,
    manifest: dict,
    answers: list[dict],
    remap: list[tuple[str, str]],
    workers: int,
) -> None:
    if (directory / "benchmark_scores.json").exists() and (
        directory / "item_scores.jsonl"
    ).exists():
        return
    dataset = {
        row["task_id"]: row for row in read_jsonl(locate(manifest["dataset"]["path"], remap))
    }
    payloads = []
    for row in answers:
        task = dataset[row["task_id"]]
        payloads.append(
            json.dumps(
                {
                    "prompt": task["prompt"],
                    "completion": row["response"],
                    "test": task["test"],
                    "entry_point": task["entry_point"],
                }
            )
        )
    probe = _humaneval_one(
        json.dumps(
            {
                "prompt": "def f():\n",
                "completion": "    return 1\n",
                "test": "def check(f):\n    assert f() == 1\n",
                "entry_point": "f",
            }
        )
    )
    if not probe:
        raise RuntimeError("Bubblewrap sandbox unavailable; HumanEval was not scored")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(_humaneval_one, payloads, chunksize=8))
    items = []
    grouped: dict[tuple, list[bool]] = defaultdict(list)
    for row, passed in zip(answers, verdicts, strict=True):
        items.append(
            {
                "key": row["key"],
                "condition": row["condition"],
                "scale": row["scale"],
                "passed": passed,
            }
        )
        grouped[(row["condition"], float(row["scale"]))].append(passed)
    write_jsonl(directory / "item_scores.jsonl", items)
    executor = Path(__file__).with_name("humaneval_execute_one.py").resolve()
    metrics = {}
    for case in manifest["conditions"]:
        for scale in case["scales"]:
            flags = grouped[(case["name"], float(scale))]
            metrics[f"{case['name']}:{scale}"] = {
                "pass_at_1": sum(flags) / len(flags),
                "passed": sum(flags),
                "n": len(flags),
                "confidence_intervals": {"pass_at_1": _interval(flags)},
                "executor_sha256": _sha(executor),
            }
    (directory / "benchmark_scores.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"humaneval {directory.parent.parent.name}", flush=True)


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_run(
    run: Path, output: Path, groups: dict[str, list[tuple[Path, dict, list[dict]]]]
) -> None:
    several = len(groups) > 1
    for _key, members in groups.items():
        judges = [item for item in members if item[1]["role"] == "judge"]
        benchmarks = [item for item in members if item[1]["role"] == "benchmark"]
        if len(judges) != 1:
            continue
        judge_dir, _manifest, _answers = judges[0]
        scored = [item for item in benchmarks if (item[0] / "benchmark_scores.json").exists()]
        names = [item[1]["dataset"]["name"] for item in scored]
        suffix = "-" + "-".join(names) if several and names else ""
        concept_rows = read_jsonl(judge_dir / "judge_scores.jsonl")
        write_json(
            output / f"{run.name}-concept{suffix}.json",
            [
                {
                    "condition": row["condition"],
                    "scale": row["scale"],
                    "key": row["key"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "concept_score": row["concept_score"],
                }
                for row in concept_rows
            ],
        )
        for bench_dir, manifest, answers in scored:
            flags = {
                (row["condition"], float(row["scale"]), row["key"]): row
                for row in read_jsonl(bench_dir / "item_scores.jsonl")
            }
            name = manifest["dataset"]["name"]
            rows = []
            for row in answers:
                extra = flags[(row["condition"], float(row["scale"]), row["key"])]
                saved = {
                    "condition": row["condition"],
                    "scale": row["scale"],
                    "key": row["key"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                }
                if name == "ifeval":
                    saved["prompt_strict"] = extra["prompt_strict"]
                    saved["prompt_loose"] = extra["prompt_loose"]
                else:
                    saved["task_id"] = row["task_id"]
                    saved["passed"] = extra["passed"]
                rows.append(saved)
            write_json(output / f"{run.name}-{name}.json", rows)
            report_dir = build_report(judge_dir, bench_dir, run)
            if report_dir is None:
                raise RuntimeError(f"report missing for {run.name} {name}")
            print(f"report {report_dir}", flush=True)


def rates_file(pipeline: Path, output: Path) -> None:
    rows = []
    for report_path in sorted(pipeline.glob("*/reports/*/report.json")):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        prefix, language, kind = parse_run(report_path.parents[2].name)
        for row in payload["rows"]:
            rows.append(
                {
                    "model": prefix,
                    "language": language,
                    "kind": kind,
                    "benchmark": row["benchmark_metric"],
                    **row,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "rates.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    kinds = sorted({(row["model"], row["kind"]) for row in rows})
    for prefix, kind in kinds:
        if not any(row["model"] == prefix and row["kind"] == kind for row in rows):
            continue
        paths = write_svgs(pipeline, kind, output, prefix=prefix)
        print(f"wrote {len(paths)} svgs for {prefix} {kind}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, default=Path("runs/pipeline"))
    parser.add_argument("--output", type=Path, default=Path("runs/results"))
    parser.add_argument(
        "--stages",
        default="judge,ifeval,humaneval,export",
        help="Comma-separated: judge, ifeval, humaneval, export",
    )
    parser.add_argument("--remap", action="append", default=[], help="source=target path prefix")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    stages = {part.strip() for part in args.stages.split(",") if part.strip()}
    remap = []
    for item in args.remap:
        source, target = item.split("=", 1)
        remap.append((source.rstrip("/"), target.rstrip("/")))
    for run in sorted(path for path in args.pipeline.iterdir() if path.is_dir()):
        try:
            parse_run(run.name)
        except ValueError:
            continue
        members = answer_dirs(run)
        groups: dict[str, list] = defaultdict(list)
        for directory, manifest, answers in members:
            groups[code_key(manifest)].append((directory, manifest, answers))
        feature = FEATURES[parse_run(run.name)[1]]
        for grouped in groups.values():
            for directory, manifest, answers in grouped:
                role = manifest["role"]
                name = manifest.get("dataset", {}).get("name")
                if role == "judge" and "judge" in stages:
                    score_language(directory, manifest, answers, feature)
                elif name == "ifeval" and "ifeval" in stages:
                    score_ifeval(directory, manifest, answers, remap)
                elif name == "humaneval" and "humaneval" in stages:
                    score_humaneval(directory, manifest, answers, remap, args.workers)
        if "export" in stages:
            export_run(run, args.output, groups)
    if "export" in stages:
        rates_file(args.pipeline, args.output)


if __name__ == "__main__":
    main()
