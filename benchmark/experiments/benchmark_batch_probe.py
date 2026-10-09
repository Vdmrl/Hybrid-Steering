"""Choose a benchmark batch using the real model, methods and longest prompts."""

import hashlib
import json
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from experiments.batch_probe import choose_batch
from experiments.smoke import representative_conditions
from hybrid_steering.benchmarks import load_tasks, read_plan
from hybrid_steering.extract import sha256
from hybrid_steering.runner import chat
from hybrid_steering.steering import SteeredModel, SteeringConfig, load_directions


def sync() -> None:
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def varied_prompts(tasks: list[dict], tokenizer, count: int) -> tuple[list[str], list[int]]:
    """Put long and short real prompts in every candidate batch."""
    measured = sorted(((len(tokenizer.encode(chat(tokenizer, row["prompt"], system=None,
                                                  enable_thinking=False), add_special_tokens=False)),
                        row["prompt"]) for row in tasks), key=lambda item: item[0])
    sample = []
    while measured and len(sample) < count:
        sample.append(measured.pop())
        if measured and len(sample) < count:
            sample.append(measured.pop(0))
    return [prompt for _, prompt in sample], [length for length, _ in sample]


def tune(model, tokenizer, direction_path: Path, plan_path: Path, data_root: Path,
         settings: dict) -> int:
    plan = read_plan(plan_path)
    tasks = load_tasks(plan, data_root)
    candidates = settings["candidates"]
    tokens = settings["probe_tokens"]
    limit = settings["max_memory_fraction"]
    if (not candidates or candidates != sorted(set(candidates))
            or any(type(n) is not int or n < 1 for n in candidates)
            or max(candidates) > len(tasks) or type(tokens) is not int or tokens < 16
            or not 0 < limit < 1):
        raise ValueError("Invalid benchmark batch probe settings")
    prompts, lengths = varied_prompts(tasks, tokenizer, max(candidates))
    conditions = [SteeringConfig(**row) for row in representative_conditions(plan["conditions"])]
    try:
        kernels_version = version("kernels")
    except PackageNotFoundError:
        kernels_version = None
    identity = {
        "plan_sha256": sha256(plan_path), "direction_sha256": sha256(direction_path),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "torch": torch.__version__, "kernels": kernels_version,
        "gpu_mask": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": [(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
                for i in range(torch.cuda.device_count())],
        "conditions": [row.__dict__ for row in conditions],
        "prompt_lengths": lengths, "candidates": candidates,
        "probe_tokens": tokens, "generation_limit": plan["max_new_tokens"],
        "max_memory_fraction": limit,
        "code_sha256": sha256(Path(__file__)),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = data_root / "performance" / f"benchmark-batch-{key}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved["identity"] != identity or saved["selected"] not in candidates:
            raise ValueError(f"Conflicting batch probe: {output}")
        print(f"BATCH_PROBE reuse batch={saved['selected']} {output}", flush=True)
        return saved["selected"]

    directions = load_directions(direction_path)
    steered = [SteeredModel(model, tokenizer, directions, condition) for condition in conditions]
    steered[0].generate(prompts[:1], min(tokens, 16))
    capacities = [torch.cuda.get_device_properties(i).total_memory / 2**20
                  for i in range(torch.cuda.device_count())]
    measurements = []
    for size in candidates:
        peaks, rates = [], []
        try:
            for condition in steered:
                for device in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device)
                sync()
                start = time.perf_counter()
                answers = condition.generate(prompts[:size], tokens)
                sync()
                if len(answers) != size or any(not answer.strip() for answer in answers):
                    raise ValueError("Benchmark probe returned an empty or missing answer")
                rates.append(size / (time.perf_counter() - start))
                peaks.append([torch.cuda.max_memory_reserved(i) / 2**20
                              for i in range(torch.cuda.device_count())])
            peak = [max(row[i] for row in peaks) for i in range(len(capacities))]
            record = {"batch_size": size, "status": "ok",
                      "examples_per_second": min(rates),
                      "method_examples_per_second": rates,
                      "peak_reserved_mib": [round(value) for value in peak],
                      "safe": all(used / total <= limit for used, total in zip(peak, capacities))}
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            record = {"batch_size": size, "status": "oom", "safe": False}
        measurements.append(record)
        print(f"BENCHMARK_BATCH_PROBE {record}", flush=True)
        if not record["safe"]:
            break
    # Short probes rank candidates; the full token budget validates memory.
    safe = [row for row in measurements if row["status"] == "ok" and row["safe"]]
    if not safe:
        raise RuntimeError("No safe benchmark batch")
    for candidate in sorted(safe, key=lambda row: row["examples_per_second"], reverse=True):
        size = candidate["batch_size"]
        try:
            peaks = []
            for condition in steered:
                for device in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device)
                answers = condition.generate(prompts[:size], plan["max_new_tokens"])
                sync()
                if len(answers) != size:
                    raise ValueError("Missing full-length probe answer")
                peaks.append([torch.cuda.max_memory_reserved(i) / 2**20
                              for i in range(torch.cuda.device_count())])
            candidate["full_length_safe"] = all(
                max(row[i] for row in peaks) / total <= limit
                for i, total in enumerate(capacities))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            candidate["full_length_safe"] = False
        if candidate["full_length_safe"]:
            selected = size
            break
    else:
        raise RuntimeError("No batch survived full-length validation")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity, "measurements": measurements,
                                     "selected": selected}, indent=2) + "\n")
    temporary.replace(output)
    print(f"BENCHMARK_BATCH_PROBE selected={selected} {output}", flush=True)
    return selected
