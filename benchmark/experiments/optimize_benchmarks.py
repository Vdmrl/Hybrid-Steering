"""Sweep concept strength, choose four pre-peak levels, then run full benchmarks.

Run on the GPU host. Model loading, batch tuning, OOM recovery, generation,
resume, and official scoring remain in experiments.queue and the package.
"""

import argparse
import copy
import csv
import fcntl
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from experiments.analyze_russian_rate_tp import wilson
from experiments.queue import conditions, prepared_plans, write_jsonl
from experiments.study import save_status
from hybrid_steering.benchmarks import EXPECTED_TASKS, _saved_for_condition, load_tasks, read_plan
from hybrid_steering.extract import sha256, training_prompts
from hybrid_steering.judge.io import read_jsonl
from hybrid_steering.paths import ROOT
from hybrid_steering.plans import write_unchanged
from hybrid_steering.plotting import plot_tradeoff
from hybrid_steering.steering import SteeringConfig
from hybrid_steering.study import select_levels, select_nearest_levels


def case_id(row: dict) -> tuple:
    return (row["method"], row.get("layer", -1), float(row["strength"]),
            row.get("normalize_to_full", False))


def grids(spec: dict) -> list[dict]:
    result = []
    for method in spec["methods"]:
        family = "activation_add" if method["method"] in {"residual", "residual_clamp"} else "recurrent_state"
        result.append({**method, "strengths": spec["strengths"][family]})
    cases = conditions(result)
    if len(cases) < 2:
        raise ValueError("Need at least one steering method")
    return result


def validate(spec: dict, root: Path) -> None:
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", spec[key]) for key in ("id", "concept")):
        raise ValueError("Study id and concept must be simple lowercase names")
    if spec["queue"].get("enable_thinking") is not False:
        raise ValueError("Explicit enable_thinking=false is required by the current runner")
    model = spec["queue"]["model"]
    if not spec["queue"]["models"][model].get("revision"):
        raise ValueError("Pin the model revision")
    metric = spec["concept_metric"]
    if metric["type"] not in {"language_rate", "judge"}:
        raise ValueError("Concept metric must be language_rate or judge")
    if metric["type"] == "language_rate" and not metric.get("language"):
        raise ValueError("Language-rate scoring needs a language code")
    if spec["screen"]["expected_n"] not in {40, 100} or spec["concept_evaluation"]["expected_n"] != 500:
        raise ValueError("Use 40 or 100 screen and 500 independent concept-evaluation prompts")
    datasets = {"screen": ("judge_prompts", spec["screen"]),
                "concept_evaluation": ("judge_prompts", spec["concept_evaluation"])}
    datasets.update({name: (name, settings) for name, settings in spec["benchmarks"].items()})
    if not spec["benchmarks"] or "judge_prompts" in spec["benchmarks"]:
        raise ValueError("Specify full benchmarks separately from concept evaluation")
    tasks = {}
    for name, (benchmark, settings) in datasets.items():
        expected = EXPECTED_TASKS.get(benchmark)
        counts = expected if isinstance(expected, tuple) else (expected,)
        if (benchmark != "mmlu_pro" and settings["expected_n"] not in counts
                or benchmark == "mmlu_pro" and settings["expected_n"] < 1):
            raise ValueError(f"Invalid benchmark or full task count: {benchmark}")
        if any(type(settings[key]) is not int or settings[key] < 1
               for key in ("expected_n", "batch_size", "max_new_tokens")):
            raise ValueError("Dataset count, batch and token limit must be positive integers")
        plan = {"benchmark": benchmark, "dataset": settings["dataset"],
                "dataset_sha256": settings["sha256"], "expected_n": settings["expected_n"]}
        if benchmark == "humaneval":
            # Inspect the raw dataset offline; the current HumanEval prompt module imports torch.
            path = (root / settings["dataset"]).resolve()
            if not path.is_relative_to(root) or sha256(path) != settings["sha256"]:
                raise ValueError("HumanEval path or dataset hash differs")
            with gzip.open(path, "rt") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            if len(rows) != settings["expected_n"] or len({row["task_id"] for row in rows}) != len(rows):
                raise ValueError("HumanEval dataset is incomplete or has duplicate tasks")
        else:
            tasks[name] = load_tasks(plan, root)
    screen = {row["prompt"].strip() for row in tasks["screen"]}
    full = {row["prompt"].strip() for row in tasks["concept_evaluation"]}
    if screen.intersection(full):
        raise ValueError("Screen and final concept prompts must be disjoint")
    for method in grids(spec):
        points = [{"strength": strength, "concept_mean": 0} for strength in method["strengths"]]
        (select_nearest_levels if spec.get("selection_rule") == "nearest" else select_levels)(
            points, spec["targets"])
    if spec.get("selection_rule") not in {None, "nearest"}:
        raise ValueError("Unknown selection rule")
    if spec["queue"].get("prepared_directions"):
        spec["queue"]["prepared_directions"][spec["concept"]]
    elif not spec["queue"].get("dataset"):
        raise ValueError("Provide prepared directions or a train.jsonl dataset for extraction")


def stage_config(spec: dict, stage: str, selected: list[dict] | None = None) -> dict:
    config = copy.deepcopy(spec["queue"])
    config["concepts"] = {spec["concept"]: "prepared"}
    config["plan_namespace"] = f"optimization-{spec['id']}-{stage}"
    # Concept scoring happens after complete snapshots, once per condition.
    config.pop("language_rate", None)
    if stage == "screen":
        config["grids"] = grids(spec)
        config["benchmarks"] = {"judge_prompts": spec["screen"]}
    else:
        config["grids"] = []
        for row in selected or []:
            grid = {"method": row["method"], "strengths": [row["strength"]],
                    "normalize_to_full": row.get("normalize_to_full", False)}
            if row.get("layer", -1) != -1:
                grid["layers"] = [row["layer"]]
            config["grids"].append(grid)
        config["benchmarks"] = {"judge_prompts": spec["concept_evaluation"], **spec["benchmarks"]}
    return config


def completed_answers(plan_path: Path, direction: Path, root: Path) -> list[dict] | None:
    """Read exact queue completion folders, never guess a legacy result identity."""
    marker = plan_path.with_suffix(".complete.json")
    if not marker.exists():
        return None
    plan = read_plan(plan_path)
    completion = json.loads(marker.read_text())
    if (completion["plan_sha256"] != sha256(plan_path)
            or completion["direction_sha256"] != sha256(direction)
            or completion["conditions"] != plan["conditions"]):
        raise ValueError("Queue completion marker belongs to a different plan or direction")
    tasks = load_tasks(plan, root)
    expected = {(case_id(case), task["key"]) for case in plan["conditions"] for task in tasks}
    found = {}
    for relative in completion["folders"]:
        folder = (root / relative).resolve()
        if not folder.is_relative_to(root / "results"):
            raise ValueError("Queue completion folder escapes the result root")
        identity = json.loads((folder / "manifest.json").read_text())["identity"]
        checks = {"concept": plan["concept"], "model": plan["model"],
                  "model_revision": plan["model_revision"], "benchmark": plan["benchmark"],
                  "dataset_sha256": plan["dataset_sha256"], "thinking": False,
                  "max_new_tokens": plan["max_new_tokens"], "decoding": "greedy", "system": None,
                  "direction_sha256": {direction.name: sha256(direction)}}
        if any(identity.get(key) != value for key, value in checks.items()):
            raise ValueError(f"Incompatible completed result: {folder}")
        core = {f"src/hybrid_steering/{name}.py" for name in
                ("runner", "state", "residual", "comparison", "steering", "benchmarks")}
        recorded = identity.get("code_sha256", {})
        if not core <= recorded.keys():
            raise ValueError("Saved result lacks core-code provenance")
        for name, digest in recorded.items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or sha256(path) != digest:
                raise ValueError("Saved steering code differs; use a new study id")
        for item in plan["conditions"]:
            case = SteeringConfig(**item)
            normalization = "per-head Frobenius to full-rank" if case.normalize_to_full else "none"
            if (identity["method"], identity["layer"], identity["schedule"], identity["normalization"]) != (
                    case.method, case.layer, case.schedule, normalization):
                continue
            answers = _saved_for_condition(folder, case, tasks, plan["model"])
            if set(answers) != {task["key"] for task in tasks}:
                raise ValueError(f"Incomplete saved answer keys: {case}")
            scores = {}
            if plan["benchmark"] != "judge_prompts":
                for row in read_jsonl(folder / "scores.jsonl"):
                    if (row["method"], row["layer"], float(row["strength"])) != (
                            case.method, case.layer, case.strength):
                        continue
                    if row["key"] in scores:
                        raise ValueError("Duplicate benchmark score")
                    scores[row["key"]] = row
                if set(scores) != set(answers):
                    raise ValueError("Missing benchmark scores")
            for task in tasks:
                row = answers[task["key"]]
                score = scores.get(task["key"], {})
                if score and score["response_sha256"] != hashlib.sha256(row["response"].encode()).hexdigest():
                    raise ValueError("Score refers to a different response")
                key = case_id(item), task["key"]
                if key in found:
                    raise ValueError("Multiple result folders cover the same condition")
                found[key] = {**row, **score, "normalize_to_full": case.normalize_to_full}
    if set(found) != expected:
        raise ValueError("Completion marker does not cover every condition and prompt")
    return [found[(case_id(case), task["key"])] for case in plan["conditions"] for task in tasks]


def run_stage(config: dict, folder: Path, root: Path, concept: str) -> dict[str, Path]:
    config_path = folder / "queue.json"
    write_unchanged(config_path, config)
    if not config.get("prepared_directions"):
        from experiments.queue import prepare as prepare_queue

        extraction_config = {**config, "benchmarks": {}, "gpu_smoke": False,
                             "pre_extraction_smoke": False}
        extraction_path = folder / "extract-queue.json"
        write_unchanged(extraction_path, extraction_config)
        cache, _, _, _ = prepare_queue(extraction_config, root, concept)
        count = len(training_prompts((root / config["dataset"]).parent))
        direction = (cache / "all" if config["direction_count"] == count else
                     cache / f"n{config['direction_count']}-seed{config['seed']}") / "directions.pt"
        if not direction.is_file():
            environment = dict(os.environ, GDN_DATA_ROOT=str(root))
            environment["PYTHONPATH"] = os.pathsep.join([
                str(ROOT / "src"), str(ROOT), environment.get("PYTHONPATH", "")])
            subprocess.run([sys.executable, "-m", "experiments.queue", "wait", "--config",
                            str(extraction_path), "--data-root", str(root)], cwd=ROOT,
                           env=environment, check=True)
        if not direction.is_file():
            raise RuntimeError("Direction extraction ended without a direction artifact")
        config = {**config, "prepared_directions": {concept: {
            "path": str(direction.relative_to(root)), "sha256": sha256(direction)}}}
        config_path = folder / "queue-prepared.json"
        write_unchanged(config_path, config)
    direction, plans = prepared_plans(config, root, concept)
    saved = [completed_answers(plan, direction, root) for plan in plans]
    if any(rows is None for rows in saved):
        environment = dict(os.environ, GDN_DATA_ROOT=str(root))
        environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT),
                                                   environment.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "experiments.queue", "wait", "--config",
                        str(config_path), "--data-root", str(root)], cwd=ROOT,
                       env=environment, check=True)
        saved = [completed_answers(plan, direction, root) for plan in plans]
    outputs = {}
    for plan, rows in zip(plans, saved):
        if rows is None:
            raise RuntimeError("Queue returned without a complete benchmark")
        name = read_plan(plan)["benchmark"]
        path = folder / name / "answers.jsonl"
        write_jsonl(path, rows)
        write_unchanged(path.parent / "snapshot.json", {"plan_sha256": sha256(plan),
                        "direction_sha256": sha256(direction), "answers_sha256": sha256(path)})
        outputs[name] = path
    return outputs


def score_concept(path: Path, spec: dict, root: Path, allow_judge: bool,
                  key_file: Path | None) -> list[dict]:
    metric = spec["concept_metric"]
    identity = {"answers_sha256": sha256(path), "metric": metric}
    if metric["type"] == "language_rate":
        identity.update(detector_version=version("langdetect"), seed=0,
                        evaluator_sha256=sha256(ROOT / "src/hybrid_steering/language_rate.py"))
    else:
        from hybrid_steering.judge.core import render
        identity.update(settings_sha256=sha256(ROOT / metric["settings"]),
                        system=render(spec["concept"], "judge_v4.md"),
                        transport_sha256={name: sha256(ROOT / "src/hybrid_steering/judge" / name)
                                          for name in ("openrouter.py", "concept_only.py")})
    cached = path.parent / "scored.jsonl"
    manifest = path.parent / "scoring.json"
    if cached.exists() and manifest.exists():
        record = json.loads(manifest.read_text())
        if record["identity"] != identity or record["scored_sha256"] != sha256(cached):
            raise ValueError("Concept scores belong to another snapshot or scoring protocol")
        return read_jsonl(cached)
    answers = read_jsonl(path)
    if metric["type"] == "language_rate":
        from hybrid_steering.language_rate import detect_language
        values = [int(detect_language(row["response"]) == metric["language"]) for row in answers]
        maximum = 1
    else:
        from hybrid_steering.judge.__main__ import main as judge, load_labels
        arguments = ["--input", str(path), "--output", str(path.parent), "--concept", spec["concept"],
                     "--data-root", str(root), "--settings", str(ROOT / metric["settings"])]
        if allow_judge:
            arguments += ["--run"]
        if key_file:
            arguments += ["--key-file", str(key_file)]
        judge(arguments)
        outputs = list(path.parent.glob("judge/*/*/progress.json"))
        if len(outputs) != 1 or not json.loads(outputs[0].read_text())["complete"]:
            raise RuntimeError("Judge is incomplete; resume with --allow-judge to authorize API calls")
        output = outputs[0].parent
        if json.loads((output / "input-snapshot.json").read_text())["answers_sha256"] != sha256(path):
            raise ValueError("Judge snapshot differs from saved answers")
        bindings = read_jsonl(output / "bindings.jsonl")
        labels = load_labels(output / "labels.jsonl")
        if len(bindings) != len(answers):
            raise ValueError("Incomplete Judge bindings")
        for row, binding in zip(answers, bindings):
            if (row["method"], row["layer"], row["strength"], row["key"]) != (
                    binding["method"], binding["layer"], binding["strength"], binding["key"]):
                raise ValueError("Judge bindings differ from source conditions")
        values = [labels[binding["anonymous_id"]]["concept_score"] for binding in bindings]
        maximum = 4
    scored = [{**row, "concept_raw_score": value, "concept_score": value / maximum}
              for row, value in zip(answers, values)]
    write_jsonl(cached, scored)
    write_unchanged(manifest, {"identity": identity, "scored_sha256": sha256(cached)})
    return scored


def select(spec: dict, scored: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in scored:
        grouped[case_id(row)].append(row)
    expected = {case_id(row) for row in conditions(grids(spec))}
    if set(grouped) != expected:
        raise ValueError("Screen does not cover the configured conditions")
    summaries = defaultdict(list)
    for case, rows in grouped.items():
        expected_n = spec["screen"]["expected_n"]
        if len(rows) != expected_n or len({row["key"] for row in rows}) != expected_n:
            raise ValueError(f"Every screen condition needs exactly {expected_n} unique prompts")
        method, layer, strength, normalized = case
        if method != "baseline":
            summaries[(method, layer, normalized)].append({"method": method, "layer": layer,
                "strength": strength, "normalize_to_full": normalized,
                "concept_mean": sum(row["concept_score"] for row in rows) / len(rows)})
    rule = select_nearest_levels if spec.get("selection_rule") == "nearest" else select_levels
    reports = [rule(points, spec["targets"]) for points in summaries.values()]
    return {"rule": ("distinct increasing strengths; minimum total absolute error" if rule is select_nearest_levels
                     else "earliest global peak; distinct increasing strengths; minimum total absolute error"),
            "groups": reports, "selected": [row for report in reports for row in report["selected"]]}


def summarize(spec: dict, scored: list[dict], outputs: dict[str, Path], folder: Path) -> None:
    concepts = defaultdict(list)
    for row in scored:
        concepts[case_id(row)].append(row["concept_score"])
    concept_intervals = {}
    for case, values in concepts.items():
        mean = sum(values) / len(values)
        if spec["concept_metric"]["type"] == "language_rate":
            lower, upper = wilson(sum(values), len(values))
        else:
            import numpy as np
            rng = np.random.default_rng(42)
            means = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
            lower, upper = map(float, np.quantile(means, [.025, .975]))
        concept_intervals[case] = mean, lower, upper, len(values)
    flags = {"ifeval": "ifeval_strict", "humaneval": "passed", "mmlu_pro": "mmlu_correct",
             "arc_challenge": "arc_correct"}
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    from colorsys import hsv_to_rgb
    for benchmark, flag in flags.items():
        if benchmark not in outputs:
            continue
        groups = defaultdict(list)
        for row in read_jsonl(outputs[benchmark]):
            if type(row.get(flag)) is not bool:
                raise ValueError(f"Missing binary benchmark verdict: {flag}")
            groups[case_id(row)].append(row[flag])
        if set(groups) != set(concepts):
            raise ValueError("Concept evaluation and benchmark conditions differ")
        rows, series = [], []
        for index, (case, values) in enumerate(groups.items()):
            method, layer, strength, normalized = case
            mean, low, high, n = concept_intervals[case]
            quality_low, quality_high = wilson(sum(values), len(values))
            rows.append({"method": method, "layer": layer, "strength": strength,
                         "normalization": "normalized" if normalized else "none", "concept": mean,
                         "concept_ci_low": low, "concept_ci_high": high, "concept_n": n,
                         "benchmark_score": sum(values) / len(values), "benchmark_ci_low": quality_low,
                         "benchmark_ci_high": quality_high, "benchmark_n": len(values)})
            label = ("Baseline" if method == "baseline" else "activation clamp" if method == "residual_clamp"
                     else "activation add" if method == "residual" else "recurrent state")
            if method.startswith("mamba_"):
                label = "Mamba state"
            if method.startswith("gdn_"):
                label = "GDN state"
            rank = re.search(r"rank(\d+)", method)
            if rank:
                label += f" · rank {rank[1]}"
            if "clamp" in method:
                label += " · clamp"
            if normalized:
                label += " · norm"
            if method != "baseline":
                label += f" · c={strength:g}"
            color = colormaps["tab20"](index) if len(groups) <= 20 else hsv_to_rgb(index / len(groups), .7, .75)
            series.append({"match": {"method": method, "layer": layer, "strength": strength,
                           "normalization": "normalized" if normalized else "none"},
                           "label": label, "color": to_hex(color)})
        path = folder / f"{benchmark}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        if spec.get("plot_family_curves"):
            keys = list(dict.fromkeys((row["method"], row["layer"], row["normalization"])
                                      for row in rows))
            palette = colormaps["tab10"]
            series = []
            for index, (method, layer, normalization) in enumerate(keys):
                label = "Baseline" if method == "baseline" else method.replace("_", " ")
                if layer >= 0:
                    label += f" L{layer}"
                series.append({"match": {"method": method, "layer": layer,
                                           "normalization": normalization},
                               "label": label, "color": "#555555" if method == "baseline"
                               else to_hex(palette((index - 1) % 10)),
                               "pareto": method != "baseline"})
        language = spec["concept_metric"]["type"] == "language_rate"
        view = {"input": str(path), "output": str(folder / benchmark),
                "model": spec["queue"]["model"].split("/")[-1], "concept": spec["concept"],
                "interval_note": "95% marginal CI · independent concept and benchmark prompts · thinking off",
                "x": {"columns": ["concept", "concept_ci_low", "concept_ci_high"],
                      "label": "Language rate (%)" if language else "Mean concept score (0–1)",
                      "scale": 100 if language else 1, "limits": [-.03, 1.03]},
                "y": {"columns": ["benchmark_score", "benchmark_ci_low", "benchmark_ci_high"],
                      "label": {"ifeval": "IFEval prompt-strict (%)", "humaneval": "HumanEval pass@1 (%)",
                                "mmlu_pro": "MMLU-Pro accuracy (%)", "arc_challenge": "ARC accuracy (%)"}[benchmark],
                      "scale": 100, "limits": [0, 1]}, "series": series}
        if spec.get("plot_family_curves"):
            view["pareto"] = {"enabled": True, "band": False,
                              "baseline": {"method": "baseline"}}
            view["point_labels"] = "strength"
        write_unchanged(folder / f"{benchmark}-plot.json", view)
        plot_tradeoff(view, root=folder)


def run(spec: dict, root: Path, allow_judge: bool, key_file: Path | None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
    folder = root / "results" / spec["queue"]["model"].split("/")[-1].lower() / spec["concept"] / "optimization" / spec["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_unchanged(folder / "study.json", spec)
        try:
            save_status(folder, stage="screen_gpu")
            screen = run_stage(stage_config(spec, "screen"), folder / "screen", root, spec["concept"])
            save_status(folder, stage="screen_concept_scoring")
            scored = score_concept(screen["judge_prompts"], spec, root, allow_judge, key_file)
            selection = {**select(spec, scored), "screen_sha256": sha256(screen["judge_prompts"]),
                         "scored_sha256": sha256(screen["judge_prompts"].parent / "scored.jsonl")}
            write_unchanged(folder / "selection.json", selection)
            save_status(folder, stage="full_gpu", selected=len(selection["selected"]))
            full = run_stage(stage_config(spec, "full", selection["selected"]), folder / "full", root, spec["concept"])
            save_status(folder, stage="full_concept_scoring")
            scored = score_concept(full["judge_prompts"], spec, root, allow_judge, key_file)
            summarize(spec, scored, full, folder)
            save_status(folder, stage="complete", selected=len(selection["selected"]))
            print(f"COMPLETE {folder}", flush=True)
        except Exception as error:
            save_status(folder, stage="failed", error=type(error).__name__)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--allow-judge", action="store_true", help="Authorize paid concept-only API calls")
    parser.add_argument("--key-file", type=Path)
    arguments = parser.parse_args()
    spec = json.loads(arguments.config.read_text())
    root = arguments.data_root.resolve()
    validate(spec, root)
    if arguments.mode == "plan":
        cases = conditions(grids(spec))
        print(json.dumps({"model": spec["queue"]["model"], "concept": spec["concept"],
                          "screen_conditions": len(cases),
                          "screen_answers": len(cases) * spec["screen"]["expected_n"],
                          "targets": spec["targets"], "full_benchmarks": list(spec["benchmarks"]),
                          "conditions": cases}, indent=2))
    else:
        run(spec, root, arguments.allow_judge, arguments.key_file)


if __name__ == "__main__":
    main()
