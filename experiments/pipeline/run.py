"""Config-driven Qwen GDN / Falcon Mamba benchmark and Judge pipeline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

import torch.distributed as dist
from batching import adaptive_batches
from evaluate import humaneval, ifeval
from humaneval_protocol import prompt as humaneval_prompt
from report import build_report

from hybrid_steering import Runner, gdn_layers, load_direction, load_runtime
from hybrid_steering.detect import concept_detector, is_language
from hybrid_steering.mamba import MambaRunner
from hybrid_steering.runtime import chat_prompts, write_jsonl
from hybrid_steering.scoring import choose, repetition, score, summarize
from residual import ResidualRunner

ROOT = Path(__file__).resolve().parents[2]
VALID_BENCHMARKS = {"ifeval", "humaneval"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved(config_file: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config_file.parent / path).resolve()


def dataset_path(config_file: Path, spec: dict) -> Path:
    if "path" in spec:
        return resolved(config_file, spec["path"])
    from huggingface_hub import hf_hub_download

    if not all(spec.get(key) for key in ("hf_repo", "hf_file", "revision")):
        raise ValueError("dataset needs path or hf_repo, hf_file, and pinned revision")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", spec["revision"]):
        raise ValueError("HF dataset revision must be a full commit SHA")
    return Path(
        hf_hub_download(
            spec["hf_repo"], spec["hf_file"], revision=spec["revision"], repo_type="dataset"
        )
    )


def read_lines(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def load_plan(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    benchmark = plan["bench_dataset"]
    if benchmark["name"] not in VALID_BENCHMARKS:
        raise ValueError("bench_dataset.name must be ifeval or humaneval")
    datasets = {}
    for role, spec in (("judge", plan["judge_dataset"]), ("benchmark", benchmark)):
        dataset = dataset_path(path, spec)
        if digest(dataset) != spec["sha256"]:
            raise ValueError(f"{role} dataset hash differs from config")
        data = read_lines(dataset)
        keys = [
            str(row.get("task_id", row.get("key", row.get("id", index))))
            for index, row in enumerate(data)
        ]
        if len(keys) != len(set(keys)) or any(not row.get("prompt") for row in data):
            raise ValueError(f"{role} dataset needs unique IDs and nonempty prompts")
        if (
            role == "benchmark"
            and spec["name"] == "ifeval"
            and len({row["prompt"] for row in data}) != len(data)
        ):
            raise ValueError("IFEval prompts must be unique for the official scorer")
        if (
            role == "benchmark"
            and spec["name"] == "humaneval"
            and any(
                not all(key in row for key in ("task_id", "test", "entry_point")) for row in data
            )
        ):
            raise ValueError("HumanEval rows need task_id, test, and entry_point")
        for row, key in zip(data, keys, strict=True):
            row["_key"] = key
        datasets[role] = data
    if not plan.get("conditions") or plan.get("max_new_tokens", 0) < 1:
        raise ValueError("conditions and max_new_tokens are required")
    if not all(plan.get("judge", {}).get(key) for key in ("feature", "config")):
        raise ValueError("judge feature and config are required")
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
            "residual_add",
            "residual_clamp",
        }:
            raise ValueError(f"unknown method: {condition['method']}")
        if not condition.get("scales") or any(
            type(x) not in (int, float) for x in condition["scales"]
        ):
            raise ValueError("every condition needs numeric scales")
        if len(set(condition["scales"])) != len(condition["scales"]) or any(
            not math.isfinite(scale) for scale in condition["scales"]
        ):
            raise ValueError("scales must be unique and finite")
        if condition["method"] == "baseline" and condition["scales"] != [0]:
            raise ValueError("baseline must use only scale 0")
        if condition["method"] != "baseline" and not condition.get("direction"):
            raise ValueError("steered conditions need a direction path")
        if condition["method"] in {"residual_add", "residual_clamp"} and (
            type(condition.get("layer")) is not int or condition["layer"] < 0
        ):
            raise ValueError("residual steering needs a non-negative integer layer")
        gain = condition.get("gain", 1)
        if type(gain) not in (int, float) or not math.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be a positive finite number")
    return plan, datasets


def identity(config_file: Path, plan: dict, data: list[dict], role: str) -> dict:
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
        "model": plan["model"],
        "model_revision": plan.get("model_revision"),
        "parallel_backend": plan.get("parallel_backend", "single"),
        "role": role,
        "dataset": plan["judge_dataset"] if role == "judge" else plan["bench_dataset"],
        "n_tasks": len(data),
        "conditions": plan["conditions"],
        "batch_size": plan.get("batch_size", 1),
        "auto_batch": plan.get("auto_batch"),
        "expression_metric": plan.get("judge", {}).get("metric"),
        "max_new_tokens": plan["max_new_tokens"],
        "thinking": False,
        "directions": directions,
        "code_sha256": {
            name: digest(ROOT / name)
            for name in (
                "experiments/pipeline/run.py",
                "experiments/pipeline/batching.py",
                "experiments/pipeline/evaluate.py",
                "experiments/pipeline/humaneval_protocol.py",
                "experiments/pipeline/residual.py",
                "src/hybrid_steering/mamba.py",
                "src/hybrid_steering/runtime.py",
                "src/hybrid_steering/runner.py",
                "src/hybrid_steering/state.py",
                "src/hybrid_steering/cache.py",
                "src/hybrid_steering/judge/steering.py",
            )
        },
    }


def run_directory(config_file: Path, plan: dict, data: list[dict], role: str, output: Path) -> Path:
    value = json.dumps(identity(config_file, plan, data, role), sort_keys=True)
    key = hashlib.sha256(value.encode()).hexdigest()[:16]
    return output / role / key


def _runner(model, tokenizer, condition: dict, config_file: Path, model_id: str):
    method = condition["method"]
    falcon = getattr(model.config, "model_type", None) == "falcon_h1"
    if (
        method not in {"baseline", "residual_add", "residual_clamp"}
        and method.startswith("mamba_") != falcon
    ):
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
    if method in {"residual_add", "residual_clamp"}:
        layer = int(condition["layer"])
        if layer not in direction:
            raise ValueError(f"residual direction has no layer {layer}")
        return ResidualRunner(
            model,
            tokenizer,
            direction[layer],
            layer,
            mode="clamp" if method == "residual_clamp" else "add",
        )
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
    if benchmark == "humaneval":
        prompts = [humaneval_prompt(stub) for stub in prompts]
    return chat_prompts(tokenizer, prompts)


def generate(
    config_file: Path, plan: dict, data: list[dict], output: Path, role: str
) -> list[dict]:
    manifest = identity(config_file, plan, data, role)
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != manifest:
            raise ValueError("existing output has a different config, dataset, direction, or code")
    elif not dist.is_initialized() or dist.get_rank() == 0:
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

            def generate_batch(batch):
                return runner.generate(
                    _texts(
                        tokenizer,
                        batch,
                        "judge" if role == "judge" else plan["bench_dataset"]["name"],
                    ),
                    scale=float(scale) * float(case.get("gain", 1)),
                    prompt_position=case.get("prompt_position", -1),
                    max_new_tokens=plan["max_new_tokens"],
                )

            def memory_fraction():
                import torch

                if not torch.cuda.is_available():
                    return 1.0
                return (
                    torch.cuda.max_memory_allocated()
                    / torch.cuda.get_device_properties(0).total_memory
                )

            def reset_memory():
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

            settings = plan.get("auto_batch", {})
            if settings and dist.is_initialized():
                raise ValueError("automatic batching is for single-GPU workers, not TP")
            missing.sort(key=lambda item: len(item["prompt"]), reverse=True)
            for batch, tokens, event in adaptive_batches(
                missing,
                width,
                generate_batch,
                maximum=settings.get("maximum", width),
                target_fraction=settings.get("memory_fraction", 0.85),
                memory_fraction=memory_fraction,
                reset_memory=reset_memory,
            ):
                if not dist.is_initialized() or dist.get_rank() == 0:
                    with (output / "batch-profile.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps({"condition": case["name"], "scale": scale, **event}) + "\n"
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
                            "gain": float(case.get("gain", 1)),
                            "key": item["_key"],
                            "prompt": item["prompt"],
                            "response": tokenizer.decode(ids, skip_special_tokens=True),
                            "repetition": repetition(ids),
                            **({"task_id": item["task_id"]} if "task_id" in item else {}),
                        }
                    )
                if dist.is_initialized():
                    hashes = [None] * dist.get_world_size()
                    dist.all_gather_object(
                        hashes, hashlib.sha256(json.dumps(additions).encode()).hexdigest()
                    )
                    if len(set(hashes)) != 1:
                        raise ValueError("TP ranks disagree on generated answers")
                if not dist.is_initialized() or dist.get_rank() == 0:
                    with answer_file.open("a", encoding="utf-8") as stream:
                        for row in additions:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if dist.is_initialized():
                    dist.barrier()
                saved.extend(additions)
                by_key.update(
                    {(row["condition"], row["scale"], row["key"]): row for row in additions}
                )
    return saved


def _score_language(answers: list[dict], output: Path, plan: dict) -> bool:
    """Concept hit from Lingua. Language runs do not call the judge model."""
    detector = concept_detector(plan["judge"]["feature"])
    rated = []
    for row in answers:
        concept = int(detector.detects(row["response"]))
        rated.append({**row, "question": row["prompt"], "concept_score": concept, "hit": concept})
    write_jsonl(output / "judge_scores.jsonl", rated)
    (output / "judge_score_manifest.json").write_text(
        json.dumps(
            {"answers": digest(output / "answers.jsonl"), "feature": plan["judge"]["feature"]}
        )
        + "\n"
    )
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
    return True


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
    config_file: Path, plan: dict, data: list[dict], output: Path, role: str, *, run_judge: bool
) -> bool:
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
    if json.loads((output / "manifest.json").read_text()) != identity(
        config_file, plan, data, role
    ):
        raise ValueError("score manifest differs from current sources")
    name = plan["bench_dataset"]["name"] if role == "benchmark" else "judge"
    benchmark_scores = output / "benchmark_scores.json"
    benchmark_manifest = output / "benchmark_score_manifest.json"
    answers_sha = digest(output / "answers.jsonl")
    benchmark_score_id = None
    if role == "benchmark":
        benchmark_score_id = {
            "answers": answers_sha,
            "scorer": digest(ROOT / "experiments/pipeline/evaluate.py"),
            "evaluator": (
                digest(ROOT / "experiments/pipeline/humaneval_execute_one.py")
                if name == "humaneval"
                else plan["bench_dataset"]["evaluator"]["evaluation_lib_sha256"]
            ),
        }
    benchmark_cached = (
        benchmark_scores.exists()
        and benchmark_manifest.exists()
        and json.loads(benchmark_manifest.read_text()) == benchmark_score_id
    )
    if role == "benchmark" and not benchmark_cached:
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
                        plan["bench_dataset"]["evaluator"],
                        dataset_path(config_file, plan["bench_dataset"]),
                    )
                else:
                    metrics[f"{case['name']}:{scale}"] = humaneval(selected, data)
        benchmark_scores.write_text(json.dumps(metrics, indent=2) + "\n")
        benchmark_manifest.write_text(json.dumps(benchmark_score_id) + "\n")
    if role != "judge":
        return True
    if is_language(plan["judge"]["feature"]) and plan["judge"].get("metric") != "judge_expression":
        return _score_language(answers, output, plan)
    prepare_judge(answers, output)
    judge = plan["judge"]
    settings = resolved(config_file, judge["config"])
    score_id = {
        "answers": digest(output / "answers.jsonl"),
        "settings": digest(settings),
        "features": digest(ROOT / "concepts/features.yaml"),
        "feature": judge["feature"],
        "judge_code": digest(ROOT / "src/hybrid_steering/judge/steering.py"),
    }
    scores_file, score_manifest = (
        output / "judge_scores.jsonl",
        output / "judge_score_manifest.json",
    )
    cached = (
        scores_file.exists()
        and score_manifest.exists()
        and json.loads(score_manifest.read_text()) == score_id
    )
    if not run_judge and not cached:
        return False
    if cached:
        rated = read_lines(scores_file)
    else:
        rated = [{**row, "question": row["prompt"]} for row in answers]
        if judge.get("metric") == "judge_expression":
            from hybrid_steering.judge import score_steering

            judgments = score_steering(
                [(r["question"], r["response"]) for r in rated],
                judge["feature"],
                batch_size=judge.get("batch_size", 8),
                settings_path=settings,
            )
            for row, judgment in zip(rated, judgments, strict=True):
                row.update(
                    concept_score=judgment.concept_score if judgment else None,
                    content_quality=judgment.content_quality if judgment else None,
                    evaluable=judgment.evaluable if judgment else None,
                    flags=list(judgment.flags) if judgment else [],
                    hit=0
                    if judgment and not judgment.evaluable
                    else int(judgment.concept_score >= 2)
                    if judgment
                    else None,
                )
        else:
            score(rated, judge["feature"], judge.get("batch_size", 8), settings_path=settings)
        write_jsonl(scores_file, rated)
        if any(
            row.get("concept_score") is None
            and not (judge.get("metric") == "judge_expression" and row.get("evaluable") is False)
            for row in rated
        ):
            raise ValueError("Judge returned incomplete scores; inspect judge_scores.jsonl")
        score_manifest.write_text(json.dumps(score_id, indent=2) + "\n")
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
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "score", "run"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-judge", action="store_true", help="Allow paid Judge API requests")
    parser.add_argument("--role", choices=("judge", "benchmark"))
    args = parser.parse_args(argv)
    config_file, output = args.config.resolve(), args.output.resolve()
    plan, datasets = load_plan(config_file)
    if args.role:
        datasets = {args.role: datasets[args.role]}
    directories = {
        role: run_directory(config_file, plan, data, role, output)
        for role, data in datasets.items()
    }
    scored = {}
    for role, data in datasets.items():
        directory = directories[role]
        if args.mode in {"generate", "run"}:
            generate(config_file, plan, data, directory, role)
        if args.mode in {"score", "run"}:
            scored[role] = evaluate(
                config_file, plan, data, directory, role, run_judge=args.run_judge
            )
    if scored.get("judge") and scored.get("benchmark"):
        build_report(directories["judge"], directories["benchmark"], output)


if __name__ == "__main__":
    main()
