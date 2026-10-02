"""Config-driven Qwen GDN / Falcon Mamba benchmark and Judge pipeline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from evaluate import humaneval, ifeval

from hybrid_steering import Runner, gdn_layers, load_direction, load_runtime
from hybrid_steering.mamba import MambaRunner
from hybrid_steering.runtime import batched, chat_prompts, write_jsonl
from hybrid_steering.scoring import choose, repetition, score, summarize

ROOT = Path(__file__).resolve().parents[2]
VALID_BENCHMARKS = {"ifeval", "humaneval", "judge_prompts"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved(config_file: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config_file.parent / path).resolve()


def read_lines(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def load_plan(path: Path) -> tuple[dict, list[dict]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    benchmark = plan["benchmark"]
    if benchmark["name"] not in VALID_BENCHMARKS:
        raise ValueError("benchmark must be ifeval, humaneval, or judge_prompts")
    dataset = resolved(path, benchmark["path"])
    if digest(dataset) != benchmark["sha256"]:
        raise ValueError("dataset hash differs from config")
    data = read_lines(dataset)
    keys = [
        str(row.get("task_id", row.get("key", row.get("id", index))))
        for index, row in enumerate(data)
    ]
    if len(keys) != len(set(keys)) or any(not row.get("prompt") for row in data):
        raise ValueError("dataset needs unique IDs and nonempty prompts")
    if benchmark["name"] == "humaneval" and any(
        not all(key in row for key in ("task_id", "test", "entry_point")) for row in data
    ):
        raise ValueError("HumanEval rows need task_id, test, and entry_point")
    for row, key in zip(data, keys, strict=True):
        row["_key"] = key
    if not plan.get("conditions") or plan.get("max_new_tokens", 0) < 1:
        raise ValueError("conditions and max_new_tokens are required")
    names = [condition["name"] for condition in plan["conditions"]]
    if len(names) != len(set(names)):
        raise ValueError("condition names must be unique")
    if sum(condition["method"] == "baseline" for condition in plan["conditions"]) > 1:
        raise ValueError("only one baseline condition is supported")
    if any(
        condition["name"] == "baseline" and condition["method"] != "baseline"
        for condition in plan["conditions"]
    ):
        raise ValueError("baseline is a reserved condition name")
    for condition in plan["conditions"]:
        if condition["method"] not in {
            "baseline",
            "gdn_add",
            "gdn_clamp",
            "mamba_add",
            "mamba_clamp",
        }:
            raise ValueError(f"unknown method: {condition['method']}")
        if not condition.get("scales") or any(
            type(x) not in (int, float) for x in condition["scales"]
        ):
            raise ValueError("every condition needs numeric scales")
        if condition["method"] == "baseline" and condition["scales"] != [0]:
            raise ValueError("baseline must use only scale 0")
        if condition["method"] != "baseline" and not condition.get("direction"):
            raise ValueError("steered conditions need a direction path")
    return plan, data


def identity(config_file: Path, plan: dict, data: list[dict]) -> dict:
    directions = {}
    for condition in plan["conditions"]:
        if condition.get("direction"):
            path = resolved(config_file, condition["direction"])
            files = (
                [path / "direction.json", path / "direction.safetensors"]
                if path.is_dir()
                else [path]
            )
            directions[str(path)] = {file.name: digest(file) for file in files}
    return {
        "config_sha256": digest(config_file),
        "model": plan["model"],
        "benchmark": plan["benchmark"],
        "n_tasks": len(data),
        "directions": directions,
        "judge_sha256": (
            {
                "settings": digest(resolved(config_file, plan["judge"]["config"])),
                "features": digest(ROOT / "concepts/features.yaml"),
            }
            if plan.get("judge")
            else None
        ),
        "code_sha256": {
            name: digest(ROOT / name)
            for name in (
                "experiments/pipeline/run.py",
                "experiments/pipeline/evaluate.py",
                "src/hybrid_steering/mamba.py",
                "src/hybrid_steering/runner.py",
                "src/hybrid_steering/state.py",
                "src/hybrid_steering/cache.py",
                "src/hybrid_steering/judge/steering.py",
            )
        },
    }


def _runner(model, tokenizer, condition: dict, config_file: Path, model_id: str):
    method = condition["method"]
    falcon = getattr(model.config, "model_type", None) == "falcon_h1"
    if method.startswith("mamba_") != falcon and method != "baseline":
        raise ValueError("Mamba methods need Falcon-H1; GDN methods need Qwen3.5")
    if method == "baseline":
        return (
            MambaRunner(model, tokenizer)
            if falcon
            else Runner(model, tokenizer, gdn_layers(model), normalize=False)
        )
    direction, manifest, target, _ = load_direction(resolved(config_file, condition["direction"]))
    if manifest.model_id not in {"unknown", model_id}:
        raise ValueError("direction model differs from config")
    rank = condition.get("rank")
    normalize = condition.get("normalize", False)
    intervention = "clamp" if method.endswith("clamp") else "add"
    if falcon:
        return MambaRunner(
            model,
            tokenizer,
            direction,
            rank=rank,
            normalize=normalize,
            intervention=intervention,
            mean_target=target,
        )
    return Runner.from_direction(
        model, tokenizer, direction, rank=rank, normalize=normalize, intervention=intervention
    )


def _texts(tokenizer, data: list[dict], benchmark: str) -> list[str]:
    prompts = [row["prompt"] for row in data]
    return prompts if benchmark == "humaneval" else chat_prompts(tokenizer, prompts)


def generate(config_file: Path, plan: dict, data: list[dict], output: Path) -> list[dict]:
    manifest = identity(config_file, plan, data)
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != manifest:
            raise ValueError("existing output has a different config, dataset, direction, or code")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    answer_file = output / "answers.jsonl"
    saved = read_lines(answer_file) if answer_file.exists() and answer_file.stat().st_size else []
    by_key = {(row["condition"], row["scale"], row["key"]): row for row in saved}
    if len(by_key) != len(saved):
        raise ValueError("duplicate saved answer key")
    expected = {
        (case["name"], float(scale), item["_key"]): item["prompt"]
        for case in plan["conditions"]
        for scale in case["scales"]
        for item in data
    }
    if any(key not in expected or expected[key] != row["prompt"] for key, row in by_key.items()):
        raise ValueError("saved answers do not match this prompt grid")
    if len(by_key) == len(expected):
        return saved
    model, tokenizer = load_runtime(plan["model"])
    family = getattr(model.config, "model_type", None)
    if family not in {"falcon_h1", "qwen3_5", "qwen3_5_text"} and plan["model"] != "tiny":
        raise ValueError(f"unsupported model family: {family}")
    width = plan.get("batch_size", 1)
    if not isinstance(width, int) or width < 1:
        raise ValueError("batch_size must be positive")
    for case in plan["conditions"]:
        runner = _runner(model, tokenizer, case, config_file, plan["model"])
        for scale in case["scales"]:
            missing = [
                item for item in data if (case["name"], float(scale), item["_key"]) not in by_key
            ]
            for batch in batched(missing, width):
                tokens = runner.generate(
                    _texts(tokenizer, batch, plan["benchmark"]["name"]),
                    scale=float(scale),
                    prompt_position=case.get("prompt_position", -1),
                    max_new_tokens=plan["max_new_tokens"],
                )
                additions = []
                for item, token_row in zip(batch, tokens, strict=True):
                    ids = [
                        int(token) for token in token_row if int(token) != tokenizer.pad_token_id
                    ]
                    additions.append(
                        {
                            "condition": case["name"],
                            "method": "baseline" if case["method"] == "baseline" else case["name"],
                            "steering_method": case["method"],
                            "scale": float(scale),
                            "key": item["_key"],
                            "prompt": item["prompt"],
                            "response": tokenizer.decode(ids, skip_special_tokens=True),
                            "repetition": repetition(ids),
                            **({"task_id": item["task_id"]} if "task_id" in item else {}),
                        }
                    )
                with answer_file.open("a", encoding="utf-8") as stream:
                    for row in additions:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                saved.extend(additions)
                by_key.update(
                    {(row["condition"], row["scale"], row["key"]): row for row in additions}
                )
    return saved


def prepare_judge(rows: list[dict], output: Path) -> None:
    tasks, bindings = {}, []
    for row in rows:
        anonymous = hashlib.sha256(
            json.dumps([row["prompt"], row["response"]], ensure_ascii=False).encode()
        ).hexdigest()
        tasks[anonymous] = {"id": anonymous, "prompt": row["prompt"], "response": row["response"]}
        bindings.append(
            {
                "id": anonymous,
                "condition": row["condition"],
                "scale": row["scale"],
                "key": row["key"],
            }
        )
    write_jsonl(output / "judge_tasks.jsonl", tasks.values())
    write_jsonl(output / "judge_bindings.jsonl", bindings)


def evaluate(
    config_file: Path, plan: dict, data: list[dict], output: Path, *, run_judge: bool
) -> None:
    answers = read_lines(output / "answers.jsonl")
    expected = len(data) * sum(len(case["scales"]) for case in plan["conditions"])
    if len(answers) != expected:
        raise ValueError("score requires the complete answer grid")
    keys = {(row["condition"], row["scale"], row["key"]) for row in answers}
    wanted = {
        (case["name"], float(scale), item["_key"])
        for case in plan["conditions"]
        for scale in case["scales"]
        for item in data
    }
    if keys != wanted:
        raise ValueError("saved answers do not match the configured grid")
    if json.loads((output / "manifest.json").read_text()) != identity(config_file, plan, data):
        raise ValueError("score manifest differs from current sources")
    name = plan["benchmark"]["name"]
    metrics = {}
    for case in plan["conditions"]:
        for scale in case["scales"]:
            selected = [
                row
                for row in answers
                if row["condition"] == case["name"] and row["scale"] == float(scale)
            ]
            if name == "ifeval":
                metrics[f"{case['name']}:{scale}"] = ifeval(
                    selected,
                    data,
                    plan["benchmark"]["evaluator"],
                    resolved(config_file, plan["benchmark"]["path"]),
                )
            elif name == "humaneval":
                metrics[f"{case['name']}:{scale}"] = humaneval(selected, data)
    if metrics:
        (output / "benchmark_scores.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if not plan.get("judge"):
        return
    prepare_judge(answers, output)
    if not run_judge:
        return
    judge = plan["judge"]
    settings = resolved(config_file, judge["config"])
    rated = [{**row, "question": row["prompt"]} for row in answers]
    score(rated, judge["feature"], judge.get("batch_size", 8), settings_path=settings)
    write_jsonl(output / "judge_scores.jsonl", rated)
    cells = summarize(rated)
    methods = {
        case["name"]: case.get("rank")
        for case in plan["conditions"]
        if case["method"] != "baseline"
    }
    chosen = (
        choose(cells, methods=methods)
        if any(case["method"] == "baseline" for case in plan["conditions"])
        else {}
    )

    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    (output / "sweep_summary.json").write_text(
        json.dumps(clean({"cells": cells, "chosen": chosen}), indent=2, allow_nan=False) + "\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "score", "run"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-judge", action="store_true", help="Allow paid Judge API requests")
    args = parser.parse_args(argv)
    config_file, output = args.config.resolve(), args.output.resolve()
    plan, data = load_plan(config_file)
    if args.mode in {"generate", "run"}:
        generate(config_file, plan, data, output)
    if args.mode in {"score", "run"}:
        evaluate(config_file, plan, data, output, run_judge=args.run_judge)


if __name__ == "__main__":
    main()
