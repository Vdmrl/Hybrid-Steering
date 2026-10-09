"""Generate IFEval, HumanEval, ARC and Judge-prompt answers through one runner.

One model and direction bundle are reused across conditions. Answers are saved
after every batch through result_store, so an interrupted run resumes by key.
"""

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

from hybrid_steering.extract import ROOT, sha256
from hybrid_steering.steering import SteeredModel, SteeringConfig, load_directions


EXPECTED_TASKS = {"ifeval": 541, "humaneval": 164, "arc_challenge": 299,
                  "judge_prompts": (20, 30, 40, 100, 500)}


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    expected = EXPECTED_TASKS.get(plan["benchmark"])
    if ((plan["benchmark"] == "mmlu_pro" and
         (type(plan["expected_n"]) is not int or plan["expected_n"] < 1)) or
        (plan["benchmark"] != "mmlu_pro" and
         (expected is None or plan["expected_n"] not in (
             expected if isinstance(expected, tuple) else (expected,))))):
        raise ValueError("Unexpected benchmark or task count")
    if plan.get("enable_thinking") is not False:
        raise ValueError("The shared decoder currently supports only no-thinking runs")
    if any(type(plan[key]) is not int or plan[key] < 1
           for key in ("batch_size", "max_new_tokens")):
        raise ValueError("batch_size and max_new_tokens must be positive integers")
    conditions = [SteeringConfig(**row) for row in plan["conditions"]]
    include_baseline = plan.get("include_baseline", True)
    if type(include_baseline) is not bool or not conditions:
        raise ValueError("Invalid baseline policy or empty job")
    if (len(conditions) != len(set(conditions))
            or sum(c.method == "baseline" for c in conditions) != int(include_baseline)):
        raise ValueError("Conditions must be unique and match the explicit baseline policy")
    return plan


def _dataset_path(plan: dict, data_root: Path) -> Path:
    root = data_root.resolve()
    path = (root / plan["dataset"]).resolve()
    if not path.is_relative_to(root) or sha256(path) != plan["dataset_sha256"]:
        raise ValueError("Dataset escapes data root or SHA256 differs from the plan")
    return path


def load_tasks(plan: dict, data_root: Path) -> list[dict]:
    """Return common {key, prompt} records while preserving official task data."""
    path = _dataset_path(plan, data_root)
    if plan["benchmark"] == "humaneval":
        from hybrid_steering.humaneval_benchmark import tasks
        rows = [{**row, "key": row["task_id"]} for row in tasks(path)]
    elif plan["benchmark"] == "arc_challenge":
        from hybrid_steering.arc_challenge import prompt, validate_row
        with path.open(encoding="utf-8") as file:
            rows = [validate_row(json.loads(line)) for line in file if line.strip()]
        rows = [{**row, "key": row["id"], "prompt": prompt(row)} for row in rows]
    elif plan["benchmark"] == "mmlu_pro":
        from hybrid_steering.mmlu_pro import task
        with path.open(encoding="utf-8") as file:
            rows = [task(json.loads(line)) for line in file if line.strip()]
    else:
        with path.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
        if plan["benchmark"] == "judge_prompts" and any(
                set(row) != {"key", "prompt", "category", "suite"}
                or row["suite"] not in {"shared", "concept"} for row in rows):
            raise ValueError("Invalid Judge prompt dataset")
    keys = [row["key"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    if (len(rows) != plan["expected_n"] or len(keys) != len(set(keys))
            or len(prompts) != len(set(prompts))
            or any(not isinstance(text, str) or not text for text in prompts)):
        raise ValueError("Benchmark task count, IDs or prompts are invalid")
    return rows


def _identity(plan: dict, condition: SteeringConfig, directions, model, tokenizer,
              data_root: Path) -> dict:
    if plan["benchmark"] == "humaneval":
        evaluator = [ROOT / "src/hybrid_steering/humaneval_score.py", ROOT / "src/hybrid_steering/humaneval_execute_one.py"]
        prompt_source = ROOT / "src/hybrid_steering/humaneval_benchmark.py"
    elif plan["benchmark"] == "ifeval":
        evaluator = [data_root / "cache/google_research/instruction_following_eval" / name
                     for name in ("evaluation_lib.py", "instructions.py", "instructions_registry.py",
                                  "instructions_util.py")]
        prompt_source = None
    elif plan["benchmark"] == "judge_prompts":
        evaluator = []  # Responses are labeled later, never auto-scored here.
        prompt_source = ROOT / "src/hybrid_steering/judge/dataset.py"
    elif plan["benchmark"] == "mmlu_pro":
        evaluator = [ROOT / "src/hybrid_steering/mmlu_pro.py"]
        prompt_source = evaluator[0]
    else:
        evaluator = [ROOT / "src/hybrid_steering/arc_challenge.py"]
        prompt_source = evaluator[0]
    code = {name: sha256(ROOT / name) for name in
            ("src/hybrid_steering/runner.py", "src/hybrid_steering/state.py",
             "src/hybrid_steering/residual.py", "src/hybrid_steering/comparison.py",
             "src/hybrid_steering/steering.py", "src/hybrid_steering/benchmarks.py")}
    if getattr(model.config, "model_type", None) == "falcon_h1":
        for name in ("architecture.py", "language_benchmark.py"):
            code["src/hybrid_steering/" + name] = sha256(ROOT / "src/hybrid_steering" / name)
    revision = getattr(model.config, "_commit_hash", None)
    if plan.get("model_revision") and revision != plan["model_revision"]:
        raise ValueError("Loaded model revision differs from benchmark plan")
    if (directions.concept and directions.concept != plan["concept"]
            or directions.model and directions.model != plan["model"]
            or directions.model_revision and directions.model_revision != revision):
        raise ValueError("Directions belong to another concept or model revision")
    return {
        "concept": plan["concept"], "method": condition.method,
        "model": plan["model"], "model_revision": revision,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "thinking": False, "max_new_tokens": plan["max_new_tokens"],
        "schedule": condition.schedule, "layer": condition.layer,
        "normalization": "per-head Frobenius to full-rank" if condition.normalize_to_full else "none",
        "direction_sha256": directions.sha256,
        "benchmark": plan["benchmark"], "dataset_sha256": plan["dataset_sha256"],
        "benchmark_protocol": "mmlu-pro-zero-shot-answer-v1" if plan["benchmark"] == "mmlu_pro" else None,
        "prompt_template_sha256": sha256(prompt_source) if prompt_source else None,
        "evaluator_sha256": {path.name: sha256(path) for path in evaluator},
        "code_sha256": code, "decoding": "greedy", "system": None,
    }


def _saved_for_condition(folder: Path, condition: SteeringConfig,
                         tasks: list[dict], model_id: str) -> dict:
    from hybrid_steering.judge.io import read_jsonl

    prompts = {row["key"]: row["prompt"] for row in tasks}
    answer_path = folder / "answers.jsonl"
    saved = read_jsonl(answer_path) if answer_path.exists() else []
    keys = [(row["method"], row["layer"], float(row["strength"]), row["key"])
            for row in saved]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate saved answers in {folder}")
    if any(prompts.get(row["key"]) != row["prompt"] or row["model"] != model_id
           or row["thinking"] is not False or not isinstance(row["response"], str)
           or row["method"] != condition.method or row["layer"] != condition.layer
           for row in saved):
        raise ValueError(f"Saved answers have incompatible prompts or protocol: {folder}")
    return {row["key"]: row for row in saved
            if float(row["strength"]) == condition.strength}


def _score(plan: dict, folder: Path, condition: SteeringConfig,
           tasks: list[dict], answers: dict, data_root: Path) -> None:
    if plan["benchmark"] == "judge_prompts":
        raise ValueError("Judge prompts have no automatic benchmark score")
    from hybrid_steering.language_benchmark import append_jsonl, read_jsonl

    score_path = folder / "scores.jsonl"
    old = read_jsonl(score_path)
    saved = {(row["layer"], float(row["strength"]), row["key"]): row for row in old}
    if len(saved) != len(old):
        raise ValueError(f"Duplicate scores in {score_path}")
    selected = [(condition.layer, condition.strength, row["key"]) for row in tasks]
    for key in selected:
        if key in saved and saved[key]["response_sha256"] != hashlib.sha256(
                answers[key[2]]["response"].encode()).hexdigest():
            raise ValueError(f"Saved score belongs to a different answer: {key}")
    metrics_path = folder / "metrics.csv"
    previous = []
    if metrics_path.exists():
        with metrics_path.open() as stream:
            previous = list(csv.DictReader(stream))
    if all(key in saved for key in selected) and any(
            (int(row["layer"]), float(row["strength"])) ==
            (condition.layer, condition.strength) for row in previous):
        return
    if plan["benchmark"] == "ifeval":
        from hybrid_steering.language_benchmark import ifeval_scores
        metrics, flags = ifeval_scores([answers[row["key"]] for row in tasks])
        verdicts = {row["key"]: flags[row["prompt"]] for row in tasks}
    elif plan["benchmark"] == "humaneval":
        from hybrid_steering.humaneval_score import execute
        with gzip.open(_dataset_path(plan, data_root), "rt") as file:
            problems = {row["task_id"]: row for row in map(json.loads, file)}
        verdicts = {row["key"]: execute(problems[row["key"]], answers[row["key"]]["completion"])
                    for row in tasks if (condition.layer, condition.strength, row["key"]) not in saved}
        metrics = {}
    elif plan["benchmark"] == "mmlu_pro":
        from hybrid_steering.mmlu_pro import score as score_mmlu
        verdicts = {row["key"]: score_mmlu(answers[row["key"]]["response"], row)
                    for row in tasks}
        metrics = {}
    else:
        from hybrid_steering.arc_challenge import score
        verdicts = {row["key"]: score(answers[row["key"]]["response"], row) for row in tasks}
        metrics = {}
    for row in tasks:
        key = (condition.layer, condition.strength, row["key"])
        if key in saved:
            continue
        result = {"key": row["key"], "method": condition.method, "layer": condition.layer,
                  "strength": condition.strength,
                  "response_sha256": hashlib.sha256(answers[row["key"]]["response"].encode()).hexdigest(),
                  **verdicts[row["key"]]}
        append_jsonl(score_path, [result])
        saved[key] = result
    if plan["benchmark"] == "humaneval":
        passed = sum(bool(saved[key]["passed"]) for key in selected)
        metrics = {"passed": passed, "pass_at_1": passed / len(tasks)}
    elif plan["benchmark"] == "arc_challenge":
        passed = sum(bool(saved[key]["arc_correct"]) for key in selected)
        valid = sum(bool(saved[key]["arc_format_valid"]) for key in selected)
        metrics = {"arc_correct": passed, "arc_accuracy": passed / len(tasks),
                   "arc_format_rate": valid / len(tasks)}
    elif plan["benchmark"] == "mmlu_pro":
        passed = sum(bool(saved[key]["mmlu_correct"]) for key in selected)
        valid = sum(bool(saved[key]["mmlu_format_valid"]) for key in selected)
        metrics = {"mmlu_correct": passed, "mmlu_accuracy": passed / len(tasks),
                   "mmlu_format_rate": valid / len(tasks)}
    summary = {"method": condition.method, "layer": condition.layer,
               "strength": condition.strength, "n": len(tasks), **metrics}
    previous = [row for row in previous if (int(row["layer"]), float(row["strength"])) !=
                (condition.layer, condition.strength)]
    temporary = metrics_path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary)
        writer.writeheader()
        writer.writerows([*previous, summary])
    temporary.replace(metrics_path)


def run_benchmark(config_path: Path, directions_path: Path, data_root: Path,
                  score: bool = True, model=None, tokenizer=None,
                  batch_size: int | None = None, output_root: Path | None = None) -> None:
    """Run all conditions; resume exact saved keys and score complete outputs."""
    data_root = data_root.resolve()
    output_root = (output_root or data_root / "results").resolve()
    if not output_root.is_relative_to(data_root):
        raise ValueError("Benchmark outputs must stay inside data_root")
    if Path(os.environ.get("GDN_DATA_ROOT", data_root)).resolve() != data_root:
        raise ValueError("GDN_DATA_ROOT must match data_root")
    os.environ.setdefault("GDN_DATA_ROOT", str(data_root))
    from hybrid_steering.language_benchmark import ROOT as benchmark_root, adaptive_batches, load_model
    from hybrid_steering.artifacts import location, merge

    plan = read_plan(config_path)
    if score and plan["benchmark"] == "judge_prompts":
        raise ValueError("Judge prompts need score=False; label saved answers separately")
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("Runtime batch size must be a positive integer")
    runtime_batch_size = batch_size or plan["batch_size"]
    print(f"BENCHMARK batch_size={runtime_batch_size}", flush=True)
    if plan["benchmark"] == "ifeval":
        if benchmark_root.resolve() != data_root or plan["dataset"] != "cache/ifeval/input_data.jsonl":
            raise ValueError("IFEval scorer requires the pinned dataset under GDN_DATA_ROOT")
    tasks = load_tasks(plan, data_root)
    if plan.get("model_slug"):
        from hybrid_steering.concepts import load as load_source, verify as verify_source
        source = load_source(plan["concept"], plan["model_slug"])
        if source["model"] != plan["model"]:
            raise ValueError("Catalog source uses another model")
        canonical = (data_root / source["data_directory"] / "directions").resolve()
        if directions_path.resolve() == canonical:
            verify_source(source, data_root)
    directions = load_directions(directions_path)
    if (model is None) != (tokenizer is None):
        raise ValueError("Pass both model and tokenizer, or neither")
    if model is None:
        placement = None
        if plan.get("placement_config"):
            from hybrid_steering.paths import project_file
            path = project_file(plan["placement_config"])
            placement = json.loads(path.read_text())["placement"]
        model, tokenizer = load_model(plan["model"], placement)
    if score and plan["benchmark"] == "humaneval":
        from hybrid_steering.humaneval_score import check_sandbox
        check_sandbox()
    completed = []
    for row in plan["conditions"]:
        condition = SteeringConfig(**row)
        identity = _identity(plan, condition, directions, model, tokenizer, data_root)
        folder = location(output_root, identity)
        manifest = folder / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError(f"Saved result uses another protocol: {folder}")
        if not manifest.exists() and (folder / "answers.jsonl").exists():
            raise ValueError(f"Saved answers lack a manifest: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        saved = _saved_for_condition(folder, condition, tasks, plan["model"])
        pending = [task for task in tasks if task["key"] not in saved]
        steered = SteeredModel(model, tokenizer, directions, condition) if pending else None
        for start in range(0, len(pending), runtime_batch_size):
            batch = pending[start:start + runtime_batch_size]
            output = adaptive_batches(batch, runtime_batch_size, lambda items:
                steered.generate([item["prompt"] for item in items], plan["max_new_tokens"]))
            records = []
            for task, answer in zip(batch, output, strict=True):
                record = {"key": task["key"], "prompt": task["prompt"],
                          "response": answer, "method": condition.method,
                          "layer": condition.layer, "strength": condition.strength,
                          "model": plan["model"], "thinking": False}
                if plan["benchmark"] == "humaneval":
                    from hybrid_steering.humaneval_benchmark import completion
                    record["completion"] = completion(answer)
                elif plan["benchmark"] == "judge_prompts":
                    record.update(category=task["category"], suite=task["suite"])
                records.append(record)
            merge(folder, identity, records, {"direction_files": directions.sha256,
                                               "dataset_sha256": plan["dataset_sha256"],
                                               "runtime_batch_size": runtime_batch_size})
            print(f"{condition.method} L{condition.layer} c={condition.strength:g}: "
                  f"{start + len(batch)}/{len(pending)}", flush=True)
        saved = _saved_for_condition(folder, condition, tasks, plan["model"])
        if len(saved) != len(tasks):
            raise ValueError(f"Incomplete condition: {condition}")
        completed.append((condition, folder, saved))
    if score:
        for condition, folder, saved in completed:
            _score(plan, folder, condition, tasks, saved, data_root)
