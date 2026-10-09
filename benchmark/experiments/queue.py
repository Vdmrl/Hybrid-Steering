"""Editable, resumable experiment queue; steering and scoring live in the package.

Usage: PYTHONPATH=src python -m experiments.queue plan|run|wait|submit --config config/queue.json
"""

import argparse
import fcntl
import hashlib
import json
import itertools
import os
import random
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from hybrid_steering.extract import extract_directions, sha256, training_prompts
from hybrid_steering.paths import ROOT, project_file
from hybrid_steering.plans import write_unchanged
from hybrid_steering.steering import SteeringConfig


class ModelLoadOOM(RuntimeError):
    """Selected GPU memory became insufficient while loading model weights."""


def is_gpu_oom(error: Exception) -> bool:
    if isinstance(error, ModelLoadOOM):
        return True
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return isinstance(error, torch.OutOfMemoryError)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            old = [json.loads(line) for line in stream if line.strip()]
        if old != rows:
            raise ValueError(f"Existing file differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def conditions(grids: list[dict]) -> list[dict]:
    output = [{"method": "baseline", "strength": 0}]
    for grid in grids:
        for layer in grid.get("layers", [-1]):
            for strength in grid["strengths"]:
                row = {"method": grid["method"], "strength": strength}
                if layer != -1:
                    row["layer"] = layer
                if grid.get("normalize_to_full", False):
                    row["normalize_to_full"] = True
                output.append(row)
    checked = [SteeringConfig(**row) for row in output]
    if len(checked) != len(set(checked)):
        raise ValueError("Duplicate steering conditions")
    return output


def paths(config: dict, data_root: Path, concept: str) -> tuple[Path, Path, Path]:
    source = (data_root / config["dataset"]).resolve()
    if not source.is_relative_to(data_root.resolve()) or not source.is_file():
        raise ValueError(f"Dataset must be a file below data root: {source}")
    dataset_hash = sha256(source)
    model_slug = config["model"].split("/")[-1].lower()
    # The cache lives beside the selected dataset, not in a detached global folder.
    extraction_id = fingerprint([config["concepts"][concept], config["extraction"],
                                 config["models"][config["model"]].get("revision")])
    cache = source.parent / "directions" / dataset_hash[:16] / model_slug / concept / extraction_id
    return source, cache, data_root / "plans" / "queue" / model_slug / concept


def subset(source: Path, cache: Path, count: int, seed: int) -> tuple[Path, Path | None]:
    if source.name != "train.jsonl":
        raise ValueError("Direction dataset must be named train.jsonl")
    rows = training_prompts(source.parent)
    if type(count) is not int or not 0 < count <= len(rows):
        raise ValueError(f"direction_count must be 1..{len(rows)}")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    selected = random.Random(seed).sample(rows, count)
    order = {row["id"]: index for index, row in enumerate(rows)}
    selected.sort(key=lambda row: order[row["id"]])
    folder = cache / f"n{count}-seed{seed}"
    write_jsonl(folder / "prompts" / "train.jsonl", selected)
    if count == len(rows):
        return folder / "prompts", None
    full_pairs = cache / "all" / "train-answers.jsonl"
    if not full_pairs.is_file():
        return folder / "prompts", None
    with full_pairs.open(encoding="utf-8") as stream:
        pairs = {row["id"]: row for row in (json.loads(line) for line in stream if line.strip())}
    if len(pairs) != len(rows):
        raise ValueError("Full dataset pair cache is incomplete")
    subset_pairs = folder / "pairs.jsonl"
    write_jsonl(subset_pairs, [pairs[row["id"]] for row in selected])
    return folder / "prompts", subset_pairs


def extraction_plan(config: dict, concept: str, instruction: str | dict, count: int,
                    reuse_pairs: Path | None) -> dict:
    target = instruction["instruction"] if isinstance(instruction, dict) else instruction
    plan = {"model": config["model"], "concept": concept, "instruction": target,
            "train_count": count, "pair_max_new_tokens": config["extraction"]["pair_max_new_tokens"],
            "prefix_tokens": config["extraction"]["prefix_tokens"],
            "batch_size": config["extraction"]["batch_size"],
            "extraction_batch_size": config["extraction"]["extraction_batch_size"],
            "residual_representation": "response_prefix", "residual_train_count": count,
            "residual_layers": config["extraction"]["residual_layers"],
            "gdn_methods": ["gdn_rank1", "gdn_rank5"],
            "placement_config": config["models"][config["model"]]["placement_config"],
            "enable_thinking": False, "system": None}
    if isinstance(instruction, dict) and instruction.get("source_instruction"):
        plan["source_instruction"] = instruction["source_instruction"]
    if reuse_pairs:
        plan["reuse_pairs_sha256"] = sha256(reuse_pairs)
    if config.get("seed_pairs") and count == config["direction_count"]:
        plan["seed_pairs"] = config["seed_pairs"]["path"]
        plan["seed_pairs_sha256"] = config["seed_pairs"]["sha256"]
    return plan


def benchmark_plans(config: dict, data_root: Path, concept: str,
                    plan_root: Path) -> list[Path]:
    from hybrid_steering.benchmarks import read_plan

    output = []
    for name, settings in config["benchmarks"].items():
        dataset = (data_root / settings["dataset"]).resolve()
        if not dataset.is_relative_to(data_root) or not dataset.is_file():
            raise FileNotFoundError(dataset)
        digest = sha256(dataset)
        if digest != settings["sha256"]:
            raise ValueError(f"Benchmark dataset changed: {dataset}")
        plan = {"benchmark": name, "concept": concept, "model": config["model"],
                "model_revision": config["models"][config["model"]].get("revision"),
                "placement_config": config["models"][config["model"]]["placement_config"],
                "dataset": settings["dataset"], "dataset_sha256": digest,
                "expected_n": settings["expected_n"], "batch_size": settings["batch_size"],
                "max_new_tokens": settings["max_new_tokens"], "enable_thinking": False,
                "conditions": conditions(config["grids"])}
        if settings.get("include_baseline") is False:
            plan["include_baseline"] = False
            plan["conditions"] = [row for row in plan["conditions"] if row["method"] != "baseline"]
        path = plan_root / name / f"{fingerprint(plan)}.json"
        write_unchanged(path, plan)
        read_plan(path)
        output.append(path)
    return output


def prepare(config: dict, data_root: Path, concept: str) -> tuple[Path, Path, Path | None, list[Path]]:
    source, cache, plan_root = paths(config, data_root, concept)
    rows = training_prompts(source.parent)
    count = config["direction_count"]
    requested_layers = {layer for grid in config["grids"]
                        if grid["method"] in {"residual", "residual_clamp"}
                        for layer in grid.get("layers", [])}
    if not requested_layers <= set(config["extraction"]["residual_layers"]):
        raise ValueError("Residual grid asks for layers missing from extraction.residual_layers")
    prompts, pairs = subset(source, cache, count, config["seed"])
    # Full answers are generated once and then reused for every subset.
    full_config = extraction_plan(config, concept, config["concepts"][concept], len(rows), None)
    full_path = cache / "all" / "extract.json"
    write_unchanged(full_path, full_config)
    selected_config = extraction_plan(config, concept, config["concepts"][concept], count, pairs)
    selected_path = prompts.parent / "extract.json"
    if pairs is not None or len(rows) == count:
        write_unchanged(selected_path, selected_config)
    plans = benchmark_plans(config, data_root, concept, plan_root)
    return cache, prompts, pairs, plans


def load_model(config: dict):
    import torch
    from hybrid_steering.language_benchmark import load_model as load

    profile = config["models"][config["model"]]
    placement_file = profile.get("placement_config")
    placement = json.loads(project_file(placement_file).read_text())["placement"] if placement_file else None
    try:
        model, tokenizer = load(config["model"], placement, revision=profile.get("revision"))
    except torch.OutOfMemoryError as exc:
        raise ModelLoadOOM("GPU memory changed or was insufficient during model loading") from exc
    if profile.get("revision") and getattr(model.config, "_commit_hash", None) != profile["revision"]:
        raise ValueError("Loaded model revision differs from configured revision")
    return model, tokenizer


def hf_home(config: dict, data_root: Path) -> Path:
    path = (data_root / config.get("hf_home", "cache/huggingface")).resolve()
    if not path.is_relative_to(data_root.resolve()):
        raise ValueError("hf_home must stay below data root")
    return path


def run(config: dict, data_root: Path) -> None:
    from hybrid_steering.benchmarks import run_benchmark

    data_root = data_root.resolve()
    os.environ["GDN_DATA_ROOT"] = str(data_root)
    os.environ["HF_HOME"] = str(hf_home(config, data_root))
    os.environ["XDG_CACHE_HOME"] = str(data_root / "cache")
    os.environ["TMPDIR"] = str(data_root / "tmp")
    os.environ["NLTK_DATA"] = str(data_root / "cache/nltk")
    if config.get("runtime_pair_batch_size"):
        os.environ["GDN_RUNTIME_BATCH_SIZE"] = str(config["runtime_pair_batch_size"])
    (data_root / "tmp").mkdir(parents=True, exist_ok=True)
    if config.get("prepared_directions"):
        run_prepared(config, data_root)
        return
    for concept in config["concepts"]:
        cache, prompts, pairs, plans = prepare(config, data_root, concept)
        model, tokenizer = load_model(config)
        try:
            full = cache / "all"
            source = data_root / config["dataset"]
            all_count = len(training_prompts(source.parent))
            if config.get("pre_extraction_smoke", False) and not (full / "directions.pt").exists():
                from experiments.smoke import check
                from hybrid_steering.comparison import prepare as prepare_direction
                small = cache / "preflight"
                small.mkdir(parents=True, exist_ok=True)
                smoke_config = json.loads((full / "extract.json").read_text())
                smoke_config.update(train_count=2, residual_train_count=2,
                                    pair_max_new_tokens=64, batch_size=2)
                prepare_direction(model, tokenizer, training_prompts(source.parent)[:2],
                                  smoke_config, small)
                check(plans[0], small / "directions.pt", data_root, model, tokenizer)
            if config.get("auto_pair_batch") and not (full / "directions.pt").exists():
                from experiments.batch_probe import tune
                selected = tune(model, tokenizer,
                                [row["prompt"] for row in training_prompts(source.parent)],
                                json.loads((full / "extract.json").read_text())["instruction"],
                                config["auto_pair_batch"],
                                full, sha256(source))
                os.environ["GDN_RUNTIME_BATCH_SIZE"] = str(selected)
            full_direction = extract_directions(source.parent, full / "extract.json", full, model, tokenizer)
            if config["direction_count"] != all_count:
                # The selected direction is recomputed from cached model answers.
                cache, prompts, pairs, plans = prepare(config, data_root, concept)
                output = prompts.parent
                direction = extract_directions(prompts, output / "extract.json", output,
                                               model, tokenizer, reuse_pairs=pairs)
            else:
                output = full
                direction = full_direction
            if config["direction_count"] != all_count:
                write_unchanged(output / "selection.json", {
                    "source_sha256": sha256(source), "selected_sha256": sha256(prompts / "train.jsonl"),
                    "selected_ids": [row["id"] for row in training_prompts(prompts)],
                    "seed": config["seed"], "count": config["direction_count"]})
            if config.get("gpu_smoke", False) and not (direction.parent / "smoke.json").exists():
                from experiments.smoke import check
                check(plans[0], direction, data_root, model, tokenizer)
            for plan in plans:
                print(f"RUN {concept} {plan.parent.name} {direction}", flush=True)
                batch = config.get("runtime_benchmark_batch_sizes", {}).get(
                    plan.parent.name, config.get("runtime_benchmark_batch_size"))
                if config.get("auto_benchmark_batch"):
                    from experiments.benchmark_batch_probe import tune
                    batch = tune(model, tokenizer, direction, plan, data_root,
                                 config["auto_benchmark_batch"])
                run_benchmark(plan, direction, data_root, score=plan.parent.name != "judge_prompts",
                              model=model, tokenizer=tokenizer, batch_size=batch)
                if plan.parent.name == "ifeval" and config.get("language_rate") == "ru":
                    from hybrid_steering.language_rate import score_ifeval_plan
                    score_ifeval_plan(plan, direction, data_root, model, tokenizer)
                if plan.parent.name == "judge_prompts" and config.get("language_rate") == "ru":
                    from hybrid_steering.language_rate import score_plan
                    score_plan(plan, direction, data_root, model, tokenizer)
        finally:
            del model, tokenizer
            import torch
            torch.cuda.empty_cache()


def prepared_plans(config: dict, data_root: Path, concept: str) -> tuple[Path, list[Path]]:
    source = config["prepared_directions"][concept]
    direction = (data_root / source["path"]).resolve()
    if not direction.is_relative_to(data_root.resolve()) or sha256(direction) != source["sha256"]:
        raise ValueError("Prepared direction path or hash differs")
    model_slug = config["model"].split("/")[-1].lower()
    plan_root = data_root / "plans/queue" / model_slug / concept
    if config.get("plan_namespace"):
        namespace = config["plan_namespace"]
        if not isinstance(namespace, str) or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                                                for char in namespace):
            raise ValueError("Plan namespace must be a simple lowercase name")
        plan_root = plan_root / namespace
    return direction, benchmark_plans(config, data_root, concept, plan_root)


def run_prepared(config: dict, data_root: Path) -> None:
    """Use frozen directions; keep all generation and scoring in run_benchmark."""
    from hybrid_steering.benchmarks import _identity, read_plan, run_benchmark
    from hybrid_steering.artifacts import location
    from hybrid_steering.steering import load_directions
    from experiments.smoke import check

    for concept in config["concepts"]:
        direction, plans = prepared_plans(config, data_root, concept)
        model, tokenizer = load_model(config)
        try:
            bundle = load_directions(direction)
            for path in plans:
                smoke = path.with_suffix(".smoke.json")
                if config.get("gpu_smoke", True) and not smoke.exists():
                    check(path, direction, data_root, model, tokenizer, output_path=smoke)
                plan = read_plan(path)
                batch = config.get("runtime_benchmark_batch_sizes", {}).get(
                    plan["benchmark"], config.get("runtime_benchmark_batch_size"))
                if config.get("auto_benchmark_batch"):
                    from experiments.benchmark_batch_probe import tune
                    batch = tune(model, tokenizer, direction, path, data_root,
                                 config["auto_benchmark_batch"])
                run_benchmark(path, direction, data_root,
                              score=plan["benchmark"] != "judge_prompts",
                              model=model, tokenizer=tokenizer, batch_size=batch)
                if plan["benchmark"] == "ifeval" and config.get("language_rate") == "ru":
                    from hybrid_steering.language_rate import score_ifeval_plan
                    score_ifeval_plan(path, direction, data_root, model, tokenizer)
                if plan["benchmark"] == "judge_prompts" and config.get("language_rate") == "ru":
                    from hybrid_steering.language_rate import score_plan
                    score_plan(path, direction, data_root, model, tokenizer)
                folders = [str(location(data_root / "results", _identity(
                    plan, SteeringConfig(**row), bundle, model, tokenizer, data_root
                )).relative_to(data_root)) for row in plan["conditions"]]
                write_unchanged(path.with_suffix(".complete.json"), {
                    "plan_sha256": sha256(path), "direction_sha256": sha256(direction),
                    "folders": sorted(set(folders)), "conditions": plan["conditions"],
                })
                print(f"COMPLETE {path}", flush=True)
        finally:
            del model, tokenizer
            import torch
            torch.cuda.empty_cache()


def wait_and_run(config: dict, data_root: Path) -> None:
    from experiments.gpus import available_uuids

    retries = int(os.environ.get("GDN_QUEUE_OOM_RETRIES", "0"))
    max_retries = config.get("max_oom_retries", 3)
    if type(max_retries) is not int or max_retries < 0 or retries < 0:
        raise ValueError("OOM retry counts must be nonnegative integers")
    profile = config["models"][config["model"]]
    count = profile["gpu_count"]
    pairs = config.get("gpu_pairs") or list(itertools.combinations(
        config.get("gpu_pool", config["gpu_ids"]), count))
    if not pairs or any(len(ids) != count or len(ids) != len(set(ids)) for ids in pairs):
        raise ValueError("Each GPU candidate must match model gpu_count without duplicates")
    lock_root = data_root / "gpu-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    while True:
        for ids in pairs:
            with ExitStack() as stack:
                files = [stack.enter_context((lock_root / f"gpu-{i}.lock").open("a+"))
                         for i in sorted(ids)]
                try:
                    for file in files:
                        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                uuids = available_uuids(ids, config.get("max_used_mib", 700),
                                        profile.get("min_total_mib", 0))
                if not uuids:
                    continue
                time.sleep(5)
                if available_uuids(ids, config.get("max_used_mib", 700),
                                   profile.get("min_total_mib", 0)) != uuids:
                    continue
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
                print(f"ALLOCATED physical GPUs {ids}, UUIDs {uuids}", flush=True)
                try:
                    run({**config, "gpu_ids": ids}, data_root)
                except Exception as exc:
                    if not is_gpu_oom(exc) or retries >= max_retries:
                        raise
                    retries += 1
                    os.environ["GDN_QUEUE_OOM_RETRIES"] = str(retries)
                    print(f"CUDA OOM on GPUs {ids}; requeueing attempt {retries}/{max_retries}: {exc}", flush=True)
                else:
                    return
            # CUDA device visibility is fixed after the first CUDA initialization.
            # Restart only after releasing our locks, so a new pair can be selected.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
        print(f"Waiting for an available GPU pair in {pairs}", flush=True)
        time.sleep(config.get("poll_seconds", 30))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "wait", "submit"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/queue.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config["model"] not in config["models"]:
        parser.error("Selected model has no profile")
    data_root = args.data_root.resolve()
    if args.mode == "submit":
        session = "gdn-queue-" + fingerprint([str(args.config.resolve()), str(data_root), time.time()])[:8]
        command = [sys.executable, "-m", "experiments.queue", "wait", "--config",
                   str(args.config.resolve()), "--data-root", str(data_root)]
        log = data_root / "logs" / (session + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        shell = " ".join(__import__("shlex").quote(item) for item in command)
        subprocess.run(["tmux", "new-session", "-d", "-s", session,
                        f"set -o pipefail; PYTHONPATH={__import__('shlex').quote(str(ROOT / 'src'))} {shell} 2>&1 | tee -a {__import__('shlex').quote(str(log))}"], check=True)
        print(f"tmux={session} log={log}")
    elif args.mode == "wait":
        wait_and_run(config, data_root)
    elif args.mode == "run":
        run(config, data_root)
    else:
        if config.get("prepared_directions"):
            for concept in config["concepts"]:
                direction, plans = prepared_plans(config, data_root, concept)
                print(f"{concept}: reused direction={direction}")
                for plan in plans:
                    print(f"  {plan.parent.name}: {plan}")
            return
        for concept in config["concepts"]:
            cache, prompts, _, plans = prepare(config, data_root, concept)
            print(f"{concept}: {len(training_prompts(prompts))} prompts, cache={cache}")
            for plan in plans:
                print(f"  {plan.parent.name}: {plan}")


if __name__ == "__main__":
    main()
