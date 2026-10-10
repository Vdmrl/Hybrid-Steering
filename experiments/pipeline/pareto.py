"""Select four concept levels between the existing pipeline's screen and full runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import run as pipeline
from report import build_report

from hybrid_steering import load_direction
from hybrid_steering.scoring import injected_norm, select_pareto_scales

GDN_SCALES = [0.75 + 0.25 * index for index in range(14)]
RESIDUAL_SCALES = [0.1 * index for index in range(1, 10)]
ROOT = Path(__file__).resolve().parents[2]


def _gpu_call(command: list[str], model: dict) -> None:
    backend = model.get("parallel_backend", "tp" if model["family"] == "gdn" else "layers")
    prefix = [sys.executable]
    if backend == "tp":
        prefix += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
    env = {**os.environ, "HYBRID_PARALLEL_BACKEND": backend}
    subprocess.run([*prefix, *command], check=True, cwd=ROOT, env=env)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(config: Path, value: str) -> Path:
    return pipeline.resolved(config, value)


def _dataset(config: Path, spec: dict) -> dict:
    path = _path(config, spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha(path)
    if spec.get("sha256", digest) != digest:
        raise ValueError(f"dataset missing or hash differs: {path}")
    resolved = {**spec, "path": str(path), "sha256": digest}
    if "evaluator" in spec:
        evaluator = dict(spec["evaluator"])
        root = _path(config, evaluator["root"])
        source = root / "instruction_following_eval/evaluation_lib.py"
        if not source.is_file():
            raise FileNotFoundError(source)
        evaluator["root"] = str(root)
        evaluator.setdefault("evaluation_lib_sha256", _sha(source))
        resolved["evaluator"] = evaluator
    return resolved


def _prompts(config: Path, suite: dict) -> tuple[dict, dict]:
    screen = _dataset(config, suite["screen_dataset"])
    confirm = _dataset(config, suite["confirm_dataset"])
    first = pipeline.read_lines(Path(screen["path"]))
    second = pipeline.read_lines(Path(confirm["path"]))
    for name, rows in (("screen", first), ("confirm", second)):
        if len(rows) != 50 or len({row["prompt"] for row in rows}) != 50:
            raise ValueError(f"{name} needs exactly 50 distinct prompts")
    if {row["prompt"] for row in first} & {row["prompt"] for row in second}:
        raise ValueError("screen and confirmation prompts must be disjoint")
    return screen, confirm


def _check_pairs(config: Path, concept: dict, screen: dict, confirm: dict) -> None:
    pairs = _path(config, concept["pairs"])
    if not pairs.is_file():
        raise FileNotFoundError(pairs)
    training = {
        row.get("user_prompt") or row.get("source_question") for row in pipeline.read_lines(pairs)
    }
    evaluation = {
        row["prompt"]
        for dataset in (screen, confirm)
        for row in pipeline.read_lines(Path(dataset["path"]))
    }
    if (training - {None}) & evaluation:
        raise ValueError(f"{concept['id']}: evaluation prompts overlap direction pairs")


def _directions(config: Path, model: dict, concept: dict, output: Path) -> tuple[Path, Path]:
    pairs = _path(config, concept["pairs"])
    if not pairs.is_file():
        raise FileNotFoundError(pairs)
    identity = [
        model["model"],
        model.get("revision"),
        _sha(pairs),
        concept["source"],
        concept["target"],
    ]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
    base = output / "directions" / model["id"] / concept["id"] / key
    common = [
        "--model",
        model["model"],
        "--concept",
        concept["id"],
        "--jsonl",
        str(pairs),
        "--source",
        concept["source"],
        "--target",
        concept["target"],
        "--batch-size",
        str(model.get("batch_size", 1)),
    ]
    for kind, command in (
        ("recurrent", ["-m", "hybrid_steering.direction"]),
        ("residual", [str(ROOT / "experiments/pipeline/residual.py"), "extract"]),
    ):
        target = base / kind / "direction"
        if not all(
            (target / name).is_file() for name in ("direction.json", "direction.safetensors")
        ):
            _gpu_call([*command, "--output", str(target.parent), *common], model)
    return base / "recurrent/direction", base / "residual/direction"


def _conditions(model: dict, recurrent: Path, residual: Path) -> list[dict]:
    method = "gdn" if model["family"] == "gdn" else "mamba"
    direction, *_ = load_direction(recurrent)
    full = injected_norm(direction, None)
    cases = [{"name": "baseline", "method": "baseline", "scales": [0]}]
    for name, rank, intervention in (
        ("rank1", 1, "add"),
        ("rank2", 2, "add"),
        ("clamp_rank1", 1, "clamp"),
        ("full", None, "add"),
    ):
        case = {
            "name": name,
            "method": f"{method}_{intervention}",
            "direction": str(recurrent),
            "normalize": False,
            "gain": full / max(injected_norm(direction, rank), 1e-8),
            "scales": GDN_SCALES,
        }
        if rank is not None:
            case["rank"] = rank
        if intervention == "add":
            case["prompt_position"] = -1
        cases.append(case)
    for layer in (16, 20):
        for name in ("residual_add", "residual_clamp"):
            cases.append(
                {
                    "name": f"{name}_L{layer}",
                    "method": name,
                    "direction": str(residual),
                    "layer": layer,
                    "scales": RESIDUAL_SCALES,
                }
            )
    return cases


def _plan(
    model: dict,
    concept: dict,
    judge_dataset: dict,
    benchmark: dict,
    judge_config: Path,
    cases: list[dict],
    tokens: int,
) -> dict:
    return {
        "model": model["model"],
        "judge_dataset": judge_dataset,
        "bench_dataset": benchmark,
        "batch_size": model.get(
            "screen_batch_size" if tokens == 512 else "full_batch_size",
            model.get("batch_size", 1),
        ),
        "parallel_backend": model.get(
            "parallel_backend", "tp" if model["family"] == "gdn" else "layers"
        ),
        "max_new_tokens": tokens,
        "conditions": cases,
        "judge": {"feature": concept["feature"], "config": str(judge_config), "batch_size": 8},
    }


def _write_plan(path: Path, plan: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError(f"existing plan differs: {path}")
    path.write_text(json.dumps(plan, indent=2) + "\n")


def _run_role(
    path: Path, output: Path, role: str, run_judge: bool, model: dict
) -> tuple[Path, bool]:
    plan, datasets = pipeline.load_plan(path)
    data = datasets[role]
    directory = pipeline.run_directory(path, plan, data, role, output)
    _gpu_call(
        [
            str(ROOT / "experiments/pipeline/run.py"),
            "generate",
            "--config",
            str(path),
            "--output",
            str(output),
            "--role",
            role,
        ],
        model,
    )
    scored = pipeline.evaluate(path, plan, data, directory, role, run_judge=run_judge)
    return directory, scored


def _select(scored: list[dict], cases: list[dict]) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    for row in scored:
        grouped[(row["condition"], float(row["scale"]))].append(row)
    selected, report = [], {}
    for case in cases:
        if case["method"] == "baseline":
            selected.append(copy.deepcopy(case))
            continue
        cells = []
        for scale in case["scales"]:
            rows = grouped[(case["name"], float(scale))]
            if len(rows) != 50 or len({row["key"] for row in rows}) != 50:
                raise ValueError(f"{case['name']} at {scale}: expected 50 scored answers")
            values = []
            for row in rows:
                score = row.get("concept_score")
                if type(score) is not int or not 0 <= score <= 4:
                    raise ValueError(f"{case['name']} at {scale}: incomplete Judge score")
                values.append(0 if score == 0 else 0.5 if score <= 2 else 1)
            cells.append({"scale": float(scale), "concept_rate": sum(values) / 50})
        try:
            choices = select_pareto_scales(cells)
        except ValueError as error:
            raise ValueError(f"{case['name']}: {error}") from error
        report[case["name"]] = choices
        selected.append({**case, "scales": [row["scale"] for row in choices]})
    return selected, report


def run_study(
    config: Path, suite: dict, model: dict, concept: dict, output: Path, run_judge: bool
) -> None:
    folder = output / model["id"] / concept["id"]
    screen, confirm = _prompts(config, suite)
    _check_pairs(config, concept, screen, confirm)
    benchmarks = [_dataset(config, item) for item in suite["benchmarks"]]
    if {item["name"] for item in benchmarks} != {"ifeval", "humaneval"}:
        raise ValueError("need IFEval and HumanEval benchmark datasets")
    judge_config = _path(config, suite["judge_config"])
    recurrent, residual = _directions(config, model, concept, output)
    cases = _conditions(model, recurrent, residual)
    screen_plan = folder / "screen.json"
    _write_plan(screen_plan, _plan(model, concept, screen, benchmarks[0], judge_config, cases, 512))
    screen_dir, ready = _run_role(screen_plan, folder / "screen", "judge", run_judge, model)
    if not ready:
        print(f"Judge tasks prepared at {screen_dir}; resume with --run-judge", flush=True)
        return
    choices, report = _select(pipeline.read_lines(screen_dir / "judge_scores.jsonl"), cases)
    selection = {
        "screen_scores_sha256": _sha(screen_dir / "judge_scores.jsonl"),
        "targets": [0.3, 0.5, 0.7, 0.99],
        "methods": report,
    }
    _write_plan(folder / "selection.json", selection)
    for benchmark in benchmarks:
        final_plan = folder / f"final-{benchmark['name']}.json"
        _write_plan(
            final_plan, _plan(model, concept, confirm, benchmark, judge_config, choices, 2048)
        )
        final_dir, ready = _run_role(final_plan, folder / "full", "judge", run_judge, model)
        if not ready:
            print(f"Judge tasks prepared at {final_dir}; resume with --run-judge", flush=True)
            return
        benchmark_dir, _ = _run_role(final_plan, folder / "full", "benchmark", run_judge, model)
        build_report(final_dir, benchmark_dir, folder / "full", two_tier=True)
    print(f"complete: {folder}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-judge", action="store_true", help="Allow paid Judge API requests")
    parser.add_argument(
        "--plan", action="store_true", help="Validate inputs without extraction or GPU"
    )
    args = parser.parse_args()
    config = args.config.resolve()
    suite = json.loads(config.read_text())
    output = args.output.resolve()
    if args.plan:
        screen, confirm = _prompts(config, suite)
        for item in suite["benchmarks"]:
            _dataset(config, item)
        for concept in suite["concepts"]:
            _check_pairs(config, concept, screen, confirm)
        print(
            json.dumps(
                {
                    "studies": len(suite["models"]) * len(suite["concepts"]),
                    "screen_conditions_per_study": 93,
                    "screen_answers_per_study": 93 * 50,
                    "selected_conditions_per_study": 33,
                    "targets": [0.3, 0.5, 0.7, 0.99],
                },
                indent=2,
            )
        )
        return
    for model in suite["models"]:
        if model["family"] not in {"gdn", "mamba"}:
            raise ValueError("model family must be gdn or mamba")
        for concept in suite["concepts"]:
            run_study(config, suite, model, concept, output, args.run_judge)


if __name__ == "__main__":
    main()
