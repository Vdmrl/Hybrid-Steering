"""One resumable study: GPU screen -> blind concept selection -> full IFEval -> audit.

worker runs on a GPU host; controller runs where OpenRouter is reachable.
Both use the same frozen JSON config. No benchmark score enters selection.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from experiments.queue import benchmark_plans, prepared_plans, wait_and_run, write_jsonl
from hybrid_steering.benchmarks import load_tasks, read_plan
from hybrid_steering.extract import sha256
from hybrid_steering.plans import write_unchanged
from hybrid_steering.steering import SteeringConfig
from hybrid_steering.study import compare, condition, means_by_condition, select_triplet


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_status(folder: Path, **values) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "worker-status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"at": time.time(), **values}, indent=2) + "\n")
    temporary.replace(path)


def collect(plan_path: Path, direction: dict, data_root: Path,
            accepted_code: list[dict], require_scores: bool = False) -> list[dict] | None:
    """Reuse only complete answers whose manifest, tasks and scores agree exactly."""
    plan = read_plan(plan_path)
    tasks = load_tasks(plan, data_root)
    prompts = {row["key"]: row["prompt"] for row in tasks}
    expected = {(condition(row), key) for row in plan["conditions"] for key in prompts}
    configs = {condition(row): SteeringConfig(**row) for row in plan["conditions"]}
    found = {}
    for path in (data_root / "results/9b" / plan["concept"]).rglob("manifest.json"):
        identity = json.loads(path.read_text()).get("identity", {})
        checks = {"benchmark": plan["benchmark"], "dataset_sha256": plan["dataset_sha256"],
                  "model": plan["model"], "model_revision": plan["model_revision"],
                  "direction_sha256": {"directions.pt": direction["sha256"]},
                  "thinking": False, "max_new_tokens": plan["max_new_tokens"],
                  "decoding": "greedy", "system": None, "concept": plan["concept"]}
        if any(identity.get(k) != v for k, v in checks.items()):
            continue
        if identity.get("code_sha256") not in accepted_code:
            continue
        if identity.get("tokenizer_revision") not in (None, plan["model_revision"]):
            raise ValueError(f"Unexpected tokenizer revision: {path}")
        if require_scores:
            evaluator = data_root / "cache/google_research/instruction_following_eval"
            hashes = {name: sha256(evaluator / name) for name in (
                "evaluation_lib.py", "instructions.py", "instructions_registry.py", "instructions_util.py")}
            if identity["evaluator_sha256"] != hashes:
                raise ValueError(f"Different IFEval evaluator: {path}")
        scores = {}
        for row in read_rows(path.parent / "scores.jsonl") if require_scores else []:
            key = condition(row), row["key"]
            if key in scores:
                raise ValueError("Duplicate official score")
            scores[key] = row
        for row in read_rows(path.parent / "answers.jsonl"):
            cfg = configs.get(condition(row))
            key = condition(row), row["key"]
            if cfg is None or key not in expected:
                continue
            norm = "per-head Frobenius to full-rank" if cfg.normalize_to_full else "none"
            if (identity["method"] != cfg.method or identity["layer"] != cfg.layer
                    or identity["schedule"] != cfg.schedule or identity["normalization"] != norm):
                continue
            if (row["prompt"] != prompts[row["key"]] or row["model"] != plan["model"]
                    or row["thinking"] is not False or not isinstance(row["response"], str)):
                raise ValueError(f"Invalid answer: {path} {key}")
            if require_scores:
                if key not in scores:
                    continue
                score = scores[key]
                if (score["response_sha256"] != hashlib.sha256(row["response"].encode()).hexdigest()
                        or type(score["ifeval_strict"]) is not bool):
                    raise ValueError("Official score belongs to another answer")
                row = {**row, "ifeval_strict": score["ifeval_strict"]}
            source = {"manifest": str(path.relative_to(data_root)), "sha256": sha256(path),
                      "identity": identity}
            saved = {**row, "source": source}
            if key in found and found[key] != saved:
                raise ValueError(f"Conflicting reuse candidates: {key}")
            found[key] = saved
    if len(found) != len(expected):
        print(f"COVERAGE {plan_path.name} {len(found)}/{len(expected)}", flush=True)
        return None
    return [found[(condition(row), task["key"])] for row in plan["conditions"] for task in tasks]


def study_folder(spec: dict, concept: str, data_root: Path) -> Path:
    return data_root / "results/9b" / concept / "judge-only" / spec["id"]


def queue_config(spec: dict, concept: str, stage: str, selection=None) -> dict:
    settings = spec["concepts"][concept]
    host = spec["hosts"][settings["host"]]
    config = copy.deepcopy(spec["queue"])
    config.update(gpu_ids=host["gpu_ids"], hf_home=host["hf_home"], concepts={concept: "prepared"},
                  prepared_directions={concept: settings["directions"]})
    if stage == "screen":
        config["grids"] = settings["screen_grids"]
        dataset = {**spec["screen"], **settings["screen_dataset"]}
        config["benchmarks"] = {"judge_prompts": dataset}
    else:
        reused = {tuple(row) for row in settings.get("reuse_full_conditions", [])}
        config["grids"] = []
        for row in selection["selected"]:
            if condition(row) in reused:
                continue
            grid = {"method": row["method"], "strengths": [row["strength"]]}
            if row["layer"] != -1:
                grid["layers"] = [row["layer"]]
            if row["method"].startswith("gdn_clamp_rank"):
                grid["normalize_to_full"] = True
            config["grids"].append(grid)
        config["benchmarks"] = {"ifeval": {**spec["full"],
            "include_baseline": ("baseline", -1, 0.) not in reused}}
    return config


def merge_rows(parts: list[list[dict]], methods: list[str]) -> list[dict]:
    output = {}
    for rows in parts:
        for row in rows:
            if row["method"] not in methods + ["baseline"]:
                continue
            key = condition(row), row["key"]
            if key in output:
                raise ValueError(f"Duplicate study answer: {key}")
            output[key] = row
    return [output[key] for key in sorted(output, key=str)]


def worker(spec: dict, concept: str, data_root: Path) -> None:
    settings = spec["concepts"][concept]
    folder = study_folder(spec, concept, data_root)
    folder.mkdir(parents=True, exist_ok=True)
    write_unchanged(folder / "study.json", spec)
    screen = folder / "screen/answers.jsonl"
    if not screen.exists():
        save_status(folder, stage="screen_gpu")
        config = queue_config(spec, concept, "screen")
        write_unchanged(folder / "screen-queue.json", config)
        wait_and_run(config, data_root)
        _, plans = prepared_plans(config, data_root, concept)
        seed = folder / "seed-screen.jsonl"
        if sha256(seed) != settings["seed_screen_sha256"]:
            raise ValueError("Frozen seed-screen answers changed")
        rows = collect(plans[0], settings["directions"], data_root, spec["accepted_code_sha256"])
        if rows is None:
            raise ValueError("GPU screen ended without complete output")
        write_jsonl(screen, merge_rows([read_rows(seed), rows], settings["methods"]))
    save_status(folder, stage="awaiting_selection")
    selection_path = folder / "selection.json"
    while not selection_path.exists():
        time.sleep(spec["poll_seconds"])
    selection = json.loads(selection_path.read_text())
    if selection["screen_sha256"] != sha256(screen) or selection["study_sha256"] != sha256(folder / "study.json"):
        raise ValueError("Selection belongs to another study or screen")
    selected_conditions = {condition(row) for row in selection["selected"]}
    if (len(selected_conditions) != len(settings["methods"])
            or {key[0] for key in selected_conditions} != set(settings["methods"])
            or not selected_conditions <= {condition(row) for row in read_rows(screen)}):
        raise ValueError("Selection is not covered by the frozen screen")
    full = folder / "full/answers.jsonl"
    if not full.exists():
        save_status(folder, stage="full_gpu")
        config = queue_config(spec, concept, "full", selection)
        write_unchanged(folder / "full-queue.json", config)
        wait_and_run(config, data_root)
        _, plans = prepared_plans(config, data_root, concept)
        sources = [*plans, *(data_root / p for p in settings.get("reuse_full_plans", []))]
        parts = []
        for plan in sources:
            rows = collect(plan, settings["directions"], data_root,
                           spec["accepted_code_sha256"], require_scores=True)
            if rows is None:
                raise ValueError(f"Incomplete full benchmark source: {plan}")
            parts.append(rows)
        rows = merge_rows(parts, settings["methods"])
        if len(rows) != spec["full"]["expected_n"] * (len(settings["methods"]) + 1):
            raise ValueError("Full output must include baseline and every selected method")
        write_jsonl(full, rows)
    save_status(folder, stage="gpu_complete", full_sha256=sha256(full))


def judge_scores(folder: Path, concept: str, data_root: Path, key_file: Path | None) -> list[dict]:
    rows = read_rows(folder / "answers.jsonl")
    if concept == "russian":
        from hybrid_steering.language_rate import detect_language
        return [{**row, "concept_score": int(detect_language(row["response"]) == "ru")} for row in rows]
    command = [sys.executable, "-m", "hybrid_steering.judge", "--input", str(folder / "answers.jsonl"),
               "--output", str(folder), "--concept", concept, "--data-root", str(data_root),
               "--run", "--workers", "8"]
    if key_file:
        command += ["--key-file", str(key_file)]
    for attempt in range(3):
        if subprocess.run(command, check=False).returncode == 0:
            break
        time.sleep(10)
    else:
        raise RuntimeError("Judge incomplete after three resumable attempts")
    protocols = [p for p in folder.glob("judge/*/*/manifest.json")
                 if json.loads(p.read_text())["identity"]["version"] == "common-v4-concept-only"]
    if len(protocols) != 1:
        raise ValueError("Expected exactly one frozen v4 Judge protocol")
    output = protocols[0].parent
    snapshot = json.loads((output / "input-snapshot.json").read_text())
    if snapshot["answers_sha256"] != sha256(folder / "answers.jsonl"):
        raise ValueError("Judge source snapshot changed")
    from hybrid_steering.judge.__main__ import load_labels
    labels = load_labels(output / "labels.jsonl")
    bindings = read_rows(output / "bindings.jsonl")
    if len(bindings) != len(rows):
        raise ValueError("Incomplete Judge bindings")
    result = []
    for row, binding in zip(rows, bindings):
        if (condition(row), row["key"]) != (condition(binding), binding["key"]):
            raise ValueError("Judge bindings disagree with saved answers")
        result.append({**row, "concept_score": labels[binding["anonymous_id"]]["concept_score"]})
    return result


def sync_from_host(spec: dict, concept: str, data_root: Path) -> None:
    host_name = spec["concepts"][concept]["host"]
    host = spec["hosts"][host_name]
    remote = study_folder(spec, concept, Path(host["data_root"]))
    local = study_folder(spec, concept, data_root)
    local.mkdir(parents=True, exist_ok=True)
    # No --delete: local Judge annotations never exist on the GPU host.
    subprocess.run(["rsync", "-a", "--exclude=selection.json", f"{host_name}:{remote}/", str(local) + "/"], check=True)


def controller(spec: dict, concept: str, data_root: Path, key_file: Path | None) -> None:
    settings = spec["concepts"][concept]
    folder = study_folder(spec, concept, data_root)
    while True:
        sync_from_host(spec, concept, data_root)
        if (folder / "study.json").exists() and json.loads((folder / "study.json").read_text()) != spec:
            raise ValueError("Remote study differs from local frozen config")
        status = json.loads((folder / "worker-status.json").read_text()) if (folder / "worker-status.json").exists() else {}
        if status.get("stage") == "failed":
            raise RuntimeError(f"Remote worker failed: {status}")
        screen = folder / "screen/answers.jsonl"
        selection = folder / "selection.json"
        if screen.exists() and not selection.exists():
            scored = judge_scores(screen.parent, concept, data_root, key_file)
            write_jsonl(screen.parent / "scored.jsonl", scored)
            means = means_by_condition(scored)
            selected = select_triplet(means, settings["methods"], settings["concept_tolerance"],
                                      settings["minimum_concept"], settings.get("fixed"))
            write_unchanged(selection, {"screen_sha256": sha256(screen),
                "scored_sha256": sha256(screen.parent / "scored.jsonl"),
                "study_sha256": sha256(folder / "study.json"), "selected": selected,
                "means": [{"method": k[0], "layer": k[1], "strength": k[2], "mean": v}
                          for k, v in sorted(means.items())],
                "rule": "max weakest concept; min spread; lower strengths; NO IFEval selection"})
        if selection.exists():
            host_name = settings["host"]
            remote = study_folder(spec, concept, Path(spec["hosts"][host_name]["data_root"]))
            subprocess.run(["rsync", "-a", str(selection), f"{host_name}:{remote}/selection.json"], check=True)
        full = folder / "full/answers.jsonl"
        if status.get("stage") == "gpu_complete" and full.exists():
            scored = judge_scores(full.parent, concept, data_root, key_file)
            write_jsonl(full.parent / "scored.jsonl", scored)
            groups = defaultdict(list)
            for row in scored:
                groups[condition(row)].append(row)
            selected = json.loads(selection.read_text())["selected"]
            contrasts = []
            for comparator in selected[1:]:
                result = compare(groups[condition(selected[0])], groups[condition(comparator)],
                    concept_maximum=settings["concept_maximum"], concept_tolerance=settings["concept_tolerance"],
                    quality_margin=spec["quality_margin"], alpha=spec["family_alpha"] / spec["n_contrasts"])
                contrasts.append({"a": selected[0], "b": comparator, **result})
            write_unchanged(folder / "comparison.json", {"study_sha256": sha256(folder / "study.json"),
                "scored_sha256": sha256(full.parent / "scored.jsonl"), "contrasts": contrasts,
                "scope": "fixed 500-example directions, Qwen3.5-9B, reused IFEval benchmark"})
            print(json.dumps({"concept": concept, "comparison": str(folder / "comparison.json"),
                              "contrasts": contrasts}, ensure_ascii=False), flush=True)
            return
        print(f"WAIT {concept}: {status.get('stage', 'starting')}", flush=True)
        time.sleep(spec["poll_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "controller", "export"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--plan", type=Path, help="export existing screening answers only")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    root = args.data_root.resolve()
    folder = study_folder(spec, args.concept, root)
    if args.mode == "export":
        rows = collect(args.plan, spec["concepts"][args.concept]["directions"], root,
                       spec["accepted_code_sha256"])
        if rows is None:
            raise ValueError("Seed screen is incomplete")
        write_jsonl(folder / "seed-screen.jsonl", rows)
        print(sha256(folder / "seed-screen.jsonl"))
    elif args.mode == "worker":
        import fcntl
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                worker(spec, args.concept, root)
            except Exception as error:
                save_status(folder, stage="failed", error=f"{type(error).__name__}: {error}")
                raise
    else:
        controller(spec, args.concept, root, args.key_file)


if __name__ == "__main__":
    main()
