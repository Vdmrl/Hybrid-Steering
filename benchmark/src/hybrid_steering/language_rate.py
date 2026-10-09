"""Score saved, complete Judge-prompt answers by detected Russian language.

This is a deterministic language detector, not an LLM judge of answer quality.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from hybrid_steering.artifacts import location
from hybrid_steering.benchmarks import _identity, load_tasks, read_plan
from hybrid_steering.steering import SteeringConfig, load_directions


def detect_language(text: str) -> str:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def score_rows(rows: list[dict], expected: dict) -> tuple[list[dict], dict]:
    if len(rows) != len(expected) or len({row["key"] for row in rows}) != len(rows):
        raise ValueError("Russian-rate scoring requires all unique prompt keys")
    if any(expected.get(row["key"]) != row["prompt"] for row in rows):
        raise ValueError("Saved prompts differ from the frozen Judge dataset")
    labels = [{"key": row["key"], "response_sha256": hashlib.sha256(row["response"].encode()).hexdigest(),
               "language": detect_language(row["response"])} for row in rows]
    russian = sum(row["language"] == "ru" for row in labels)
    return labels, {"n": len(rows), "russian": russian, "russian_rate": russian / len(rows),
                    "detector": "langdetect", "detector_version": version("langdetect"), "seed": 0}


def score_plan(plan_path: Path, directions_path: Path, data_root: Path, model, tokenizer) -> None:
    plan = read_plan(plan_path)
    if plan["benchmark"] != "judge_prompts":
        raise ValueError("Language rate here requires frozen Judge prompts")
    tasks = load_tasks(plan, data_root)
    expected = {row["key"]: row["prompt"] for row in tasks}
    directions = load_directions(directions_path)
    for item in plan["conditions"]:
        condition = SteeringConfig(**item)
        identity = _identity(plan, condition, directions, model, tokenizer, data_root)
        folder = location(data_root / "results", identity)
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError(f"Result provenance differs: {folder}")
        rows = [json.loads(line) for line in (folder / "answers.jsonl").read_text().splitlines() if line]
        rows = [row for row in rows if float(row["strength"]) == condition.strength]
        labels, summary = score_rows(rows, expected)
        summary.update(method=condition.method, layer=condition.layer, strength=condition.strength,
                       dataset_sha256=plan["dataset_sha256"], direction_sha256=directions.sha256)
        for path, payload in ((folder / "language-scores.json", labels),
                              (folder / "language-metrics.json", summary)):
            if path.exists() and json.loads(path.read_text()) != payload:
                raise ValueError(f"Saved language score conflicts: {path}")
            if not path.exists():
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                temporary.replace(path)
        print(f"RUSSIAN_RATE {condition.method} L{condition.layer} c={condition.strength:g} "
              f"{summary['russian']}/{summary['n']}", flush=True)


def score_ifeval_plan(plan_path: Path, directions_path: Path, data_root: Path,
                      model, tokenizer) -> None:
    """Save a per-strength Russian-rate audit of complete, officially scored IFEval answers."""
    from hybrid_steering.benchmarks import _saved_for_condition
    from hybrid_steering.judge.io import read_jsonl
    from hybrid_steering.plans import write_unchanged

    plan = read_plan(plan_path)
    if plan["benchmark"] != "ifeval" or plan["concept"] != "russian":
        raise ValueError("Expected a Russian IFEval plan")
    tasks = load_tasks(plan, data_root)
    expected = {task["key"]: task["prompt"] for task in tasks}
    directions = load_directions(directions_path)
    for item in plan["conditions"]:
        condition = SteeringConfig(**item)
        identity = _identity(plan, condition, directions, model, tokenizer, data_root)
        folder = location(data_root / "results", identity)
        manifest = folder / "manifest.json"
        if json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError(f"Result provenance differs: {folder}")
        answers = _saved_for_condition(folder, condition, tasks, plan["model"])
        scores = [row for row in read_jsonl(folder / "scores.jsonl")
                  if row["method"] == condition.method and row["layer"] == condition.layer
                  and float(row["strength"]) == condition.strength]
        keyed = {row["key"]: row for row in scores}
        if (len(answers) != len(tasks) or len(scores) != len(tasks)
                or len(keyed) != len(tasks) or set(keyed) != set(expected)):
            raise ValueError(f"Incomplete or duplicate official IFEval scores: {condition}")
        rows = [answers[task["key"]] for task in tasks]
        for row in rows:
            score = keyed[row["key"]]
            if (score["response_sha256"] != hashlib.sha256(row["response"].encode()).hexdigest()
                    or type(score["ifeval_strict"]) is not bool):
                raise ValueError(f"IFEval score belongs to another answer: {row['key']}")
        labels, summary = score_rows(rows, expected)
        summary.update(method=condition.method, layer=condition.layer, strength=condition.strength,
                       dataset_sha256=plan["dataset_sha256"], direction_sha256=directions.sha256,
                       identity_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                       answers_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                       scores_sha256=hashlib.sha256(json.dumps(
                           [keyed[task["key"]] for task in tasks], sort_keys=True).encode()).hexdigest())
        output = folder / f"russian-rate-c{condition.strength:g}.json"
        write_unchanged(output, {"summary": summary, "labels": labels})
        print(f"RUSSIAN_RATE {condition.method} L{condition.layer} c={condition.strength:g} "
              f"{summary['russian']}/{summary['n']}", flush=True)
