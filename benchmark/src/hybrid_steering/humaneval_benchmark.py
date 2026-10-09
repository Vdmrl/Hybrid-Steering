"""Run paired 9B GDN/residual steering on the pinned HumanEval tasks."""

import argparse
import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path
from hybrid_steering.paths import ROOT as PROJECT_ROOT

import torch

from hybrid_steering import language_benchmark as bench
from hybrid_steering.comparison import generate_condition
from hybrid_steering.concepts import asset_path, load as load_source, verify as verify_source
from hybrid_steering.state import analyze, match_head_norm


PROMPT = ("Solve this Python task. Return a complete implementation of the requested "
          "function, including its def line. Output Python code only: no Markdown "
          "fences, explanations, or tests.\n\n{stub}")
METHOD_FOLDERS = {"baseline": "baseline", "residual": "residual",
                  "gdn_full": "gdn-add-full", "gdn_rank5": "gdn-add-rank5",
                  "gdn_clamp_rank1": "gdn-clamp-rank1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tasks(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    ids = {row["task_id"] for row in rows}
    if len(rows) != len(ids) != 164 or len(rows) != 164:
        raise ValueError("Expected 164 unique HumanEval tasks")
    return [{"task_id": row["task_id"], "stub": row["prompt"],
             "entry_point": row["entry_point"],
             "prompt": PROMPT.format(stub=row["prompt"])} for row in rows]


def completion(response: str) -> str:
    """Only remove a single enclosing code fence; never repair generated code."""
    text = response.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    return "\n\n" + (match.group(1) if match else text) + "\n"


def cases(config: dict) -> list[tuple[str, int, float]]:
    rows = [(method, int(layer), float(strength)) for method, layer, strength in config["cases"]]
    if len(rows) != len(set(rows)) or ("baseline", -1, 0.0) not in rows:
        raise ValueError("Cases must be unique and include a true baseline")
    for method, layer, strength in rows:
        if method not in METHOD_FOLDERS or (layer == -1) != (method != "residual"):
            raise ValueError(f"Invalid method/layer: {method}, {layer}")
        if method != "baseline" and strength <= 0:
            raise ValueError("Steering strengths must be positive")
    return rows


def sources(source: dict, data_root: Path) -> tuple[dict, str]:
    verify_source(source, data_root)
    gdn = torch.load(asset_path(source, "gdn_direction", data_root), map_location="cpu", weights_only=False)
    residual = torch.load(asset_path(source, "residual_direction", data_root), map_location="cpu", weights_only=False)
    normalized, report = match_head_norm(gdn["gdn_rank1"], gdn["gdn_full"])
    if any(layer["capped_heads"] for layer in report.values()):
        raise ValueError("Rank-1 normalization reached the gain cap")
    _, factors = analyze(normalized, "cpu", rank=1)
    artifact = {"gdn_full": gdn["gdn_full"], "gdn_rank5": gdn["gdn_rank5"],
                "clamp_factors": factors, "units": residual["units"],
                "targets": residual["targets"],
                "residual_directions": residual["residual_directions"]}
    provenance = [json.loads(asset_path(source, name, data_root).read_text())
                  for name in ("gdn_provenance", "residual_provenance")]
    revisions = {row["model_revision"] for row in provenance}
    if len(revisions) != 1 or any(row["identity"]["config"]["model"] != source["model"]
                                  for row in provenance):
        raise ValueError("Direction sources have incompatible model revisions")
    return artifact, revisions.pop()


def folder(data_root: Path, config: dict, method: str) -> Path:
    return (data_root / "results" / config["concept"] / METHOD_FOLDERS[method]
            / config["model_slug"] / "humaneval-v1")


def protocol(config: dict, source: dict, model_revision: str) -> dict:
    code = {name: sha256(PROJECT_ROOT / name) for name in
            ("src/hybrid_steering/runner.py", "src/hybrid_steering/state.py",
             "src/hybrid_steering/residual.py", "src/hybrid_steering/comparison.py",
             "src/hybrid_steering/language_benchmark.py", "src/hybrid_steering/cpu_transfer.py",
             "src/hybrid_steering/humaneval_benchmark.py")}
    return {"model": source["model"], "model_revision": model_revision,
            "dataset_commit": config["dataset_commit"],
            "dataset_sha256": config["dataset_sha256"],
            "prompt_style": config["prompt_style"], "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "max_new_tokens": config["max_new_tokens"], "enable_thinking": False,
            "decoding": "greedy", "system": None, "code_sha256": code,
            "source_assets": {name: info["sha256"] for name, info in source["assets"].items()},
            "residual_schedule": "every generated token",
            "gdn_full_schedule": "once after split-prefill",
            "gdn_rank5_schedule": "once after split-prefill, unnormalized",
            "gdn_clamp_rank1_schedule": "hard clamp every generated token, per-head Frobenius-normalized"}


def existing(path: Path, method: str, task_rows: list[dict]) -> set[tuple[int, float, str]]:
    prompts = {row["task_id"]: row["prompt"] for row in task_rows}
    rows = bench.read_jsonl(path)
    keys = {(row["layer"], float(row["strength"]), row["task_id"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError(f"Duplicate answers in {path}")
    if any(row["method"] != method or row["prompt"] != prompts[row["task_id"]] for row in rows):
        raise ValueError(f"Incompatible saved answers in {path}")
    return keys


@torch.inference_mode()
def main(config_path: Path, data_root: Path, smoke: bool) -> None:
    config = json.loads(config_path.read_text())
    if config["enable_thinking"] is not False or config["prompt_style"] != "chat-full-function-v1":
        raise ValueError("This run uses the pinned no-thinking full-function protocol")
    planned = cases(config)
    source = load_source(config["concept"], config["model_slug"])
    task_path = data_root / config["dataset"]
    if sha256(task_path) != config["dataset_sha256"]:
        raise ValueError("HumanEval dataset SHA256 mismatch")
    task_rows = tasks(task_path)
    artifact, revision = sources(source, data_root)
    placement = json.loads(Path(config["placement_config"]).read_text())["placement"]
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_properties(i).total_memory < 15 * 2**30
                                             for i in range(2)):
        raise ValueError("Expected two visible 16-GB A4000 GPUs")
    model, tokenizer = bench.load_model(source["model"], placement)
    if getattr(model.config, "_commit_hash", None) != revision:
        raise ValueError("Loaded model revision differs from direction source")
    for layer, values in artifact["clamp_factors"].items():
        device = next(model.model.language_model.layers[layer].parameters()).device
        artifact["clamp_factors"][layer] = tuple(value.to(device=device, dtype=torch.bfloat16)
                                                  for value in values)
    identity = protocol(config, source, revision)
    if smoke:
        chosen = [planned[0], next(case for case in planned if case[0] == "residual"),
                  next(case for case in planned if case[0] == "gdn_full"),
                  next(case for case in planned if case[0] == "gdn_clamp_rank1")]
        sample = task_rows[:2]
        saved = []
        for method, layer, strength in chosen:
            answers = generate_condition(model, tokenizer, sample, artifact, method, layer, strength, 256)
            saved += [{"task_id": row["task_id"], "method": method, "layer": layer,
                       "strength": strength, "response": answer,
                       "completion": completion(answer),
                       "has_entry_point": bool(re.search(rf"\bdef\s+{re.escape(row['entry_point'])}\s*\(", answer))}
                      for row, answer in zip(sample, answers)]
            print(f"SMOKE {method} c={strength}: {sum(r['has_entry_point'] for r in saved[-2:])}/2 functions", flush=True)
        bench.write_jsonl(data_root / "results" / config["concept"] / "humaneval-smoke.jsonl", saved)
        return
    probe = sorted(task_rows, key=lambda row: len(row["prompt"]))[len(task_rows)//2:]
    timings = []
    for size in config["batch_candidates"]:
        started = time.monotonic()
        try:
            answers = generate_condition(model, tokenizer, probe[:size], artifact, "baseline", -1, 0.0, 256)
            elapsed = time.monotonic() - started
            timings.append({"batch_size": size, "seconds": elapsed, "answers_per_second": size/elapsed,
                            "nonempty": sum(bool(answer.strip()) for answer in answers)})
            print("PROBE", json.dumps(timings[-1]), flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    if not timings:
        raise RuntimeError("No viable generation batch size")
    batch_size = max(timings, key=lambda row: row["answers_per_second"])["batch_size"]
    print(f"SELECTED BATCH {batch_size}", flush=True)
    for method, layer, strength in planned:
        output = folder(data_root, config, method)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.json"
        manifest = {"protocol": identity, "method": method, "physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
                    "device_map": model.hf_device_map, "batch_probe": timings,
                    "runtime_batch_size": batch_size}
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["protocol"] != identity or old["method"] != method:
                raise ValueError(f"Protocol mismatch in {output}")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        saved = existing(output / "answers.jsonl", method, task_rows)
        pending = [row for row in task_rows if (layer, strength, row["task_id"]) not in saved]
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            began = time.monotonic()
            answers = bench.adaptive_batches(batch, batch_size, lambda rows:
                generate_condition(model, tokenizer, rows, artifact, method, layer, strength,
                                   config["max_new_tokens"]))
            rows = [{"task_id": item["task_id"], "prompt": item["prompt"],
                     "stub": item["stub"], "entry_point": item["entry_point"],
                     "response": answer, "completion": completion(answer),
                     "method": method, "layer": layer, "strength": strength,
                     "model": source["model"], "thinking": False}
                    for item, answer in zip(batch, answers)]
            bench.append_jsonl(output / "answers.jsonl", rows)
            print(f"{method} L{layer} c={strength} {start+len(batch)}/{len(pending)} "
                  f"seconds={time.monotonic()-began:.1f}", flush=True)
    print(f"GENERATION COMPLETE: {len(planned)} conditions x {len(task_rows)} tasks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/legacy-config/configs/9b/humaneval_russian_9b.json"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ["GDN_DATA_ROOT"]))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.data_root, args.smoke)
