"""Shard a frozen §4.4 job across independent single-GPU workers."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_job_configs import PACKAGE, read, read_rows, sha
from native_runtime import model_provenance
from run_experiment import (
    atomic_json,
    atomic_rows,
    export_judge,
    fingerprint_of,
    identity,
    opaque_id,
    output_lock,
    resolve_job,
    scheduled_tasks,
)


def worker_rows(path):
    # A worker writes, flushes and fsyncs each line. Do not import a torn tail
    # after a hard kill; leave the original file intact for diagnosis.
    with path.open("rb") as stream:
        return [json.loads(line) for line in stream if line.endswith(b"\n")]


def collect(output, tasks, namespace):
    merged, manifests = {}, {}
    root_manifest = output / "manifest.json"
    gamma = read(root_manifest).get("gamma_by_condition", {}) if root_manifest.exists() else {}
    root = output / "generations.jsonl"
    workers = sorted((output / "workers").glob("worker_*"))
    for worker in workers:
        path = worker / "manifest.json"
        raw = worker / "generations.jsonl"
        if not path.exists():
            if raw.exists():
                raise ValueError(f"Missing worker manifest: {worker}")
            continue
        manifest = read(path)
        if manifest.get("parallel_namespace") != namespace:
            raise ValueError(f"Unexpected worker provenance: {worker}")
        manifests[worker.name] = manifest
        for key, value in manifest["gamma_by_condition"].items():
            if key in gamma and gamma[key] != value:
                raise ValueError(f"Inconsistent RSS metadata: {key}")
            gamma[key] = value
        allowed = {tuple(pair) for pair in manifest["job"]["tasks"]}
        if raw.exists():
            for row in worker_rows(raw):
                key = row["condition_id"], row["question_id"]
                check_row(row, key, tasks, namespace)
                if key not in allowed or key in merged:
                    raise ValueError(f"Unexpected or duplicate worker task: {key}")
                merged[key] = row
    root_seen = set()
    if root.exists():
        for row in read_rows(root):
            key = row["condition_id"], row["question_id"]
            check_row(row, key, tasks, namespace)
            if key in root_seen or (key in merged and merged[key] != row):
                raise ValueError(f"Conflicting merged task: {key}")
            root_seen.add(key)
            merged[key] = row
    return merged, manifests, gamma


def check_row(row, key, tasks, namespace):
    if (
        key not in tasks
        or row["answer_id"] != opaque_id(namespace, key)
        or row["prompt_id"] != key[1]
        or row["prompt"] != tasks[key][1]["text"]
        or row["job"] != (tasks[key][0] if tasks[key][0]["active"] else None)
    ):
        raise ValueError(f"Unexpected generation row: {key}")


def make_plan(args, cfg, tasks, output, namespace):
    devices = args.gpus.split(",")
    slots = [gpu for gpu in devices for _ in range(args.workers_per_gpu)]
    first = 1 + max(
        (int(p.name.rsplit("_", 1)[1]) for p in (output / "workers").glob("worker_*")),
        default=-1,
    )
    mixed = len({c["method"] for c in cfg["conditions"]}) > 1
    for index, gpu in enumerate(slots):
        assigned = tasks[index :: len(slots)]
        if not assigned:
            continue
        folder = output / "workers" / f"worker_{first + index:02d}"
        needed = {key[0] for key in assigned}
        shard = {
            **cfg,
            "conditions": [c for c in cfg["conditions"] if c["id"] in needed],
            "expected_answers": len(assigned),
            "tasks": assigned,
        }
        cmd = [
            sys.executable,
            "-B",
            str(PACKAGE / "run_experiment.py"),
            "job" if mixed else args.runner,
            "--job",
            str(folder / "shard_job.json"),
            "--output",
            str(folder),
            "--parallel-namespace",
            namespace,
        ]
        if args.mock:
            cmd.append("--mock")
        else:
            cmd.extend(("--model", str(args.model.resolve())))
        yield folder, gpu, shard, cmd


def execute(plan, args):
    workers, failed = [], []

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt

    handled = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, (folder, gpu, shard, cmd) in enumerate(plan):
            if index:
                time.sleep(args.stagger_sec)
            folder.mkdir(parents=True)
            atomic_json(folder / "shard_job.json", shard)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if args.mock else gpu}
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                worker = subprocess.Popen(
                    cmd, cwd=PACKAGE, stdout=log, stderr=subprocess.STDOUT, env=env
                )
            workers.append((folder, worker))
        for folder, worker in workers:
            if worker.wait():
                failed.append(folder.name)
                with (folder / "worker.log").open(encoding="utf-8", errors="replace") as log:
                    tail = "".join(deque(log, maxlen=15))
                print(
                    f"{folder.name} exited {worker.returncode}:\n{tail}",
                    file=sys.stderr,
                )
    finally:
        for sig in handled:
            signal.signal(sig, signal.SIG_IGN)
        try:
            for _, worker in workers:
                if worker.poll() is None:
                    worker.terminate()
            deadline = time.monotonic() + 10
            for _, worker in workers:
                try:
                    worker.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
        finally:
            for sig, handler in handled.items():
                signal.signal(sig, handler)
    return failed


def launch(args, cfg, tasks, frozen, namespace):
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or read(manifest_path)["run_sha256"] != namespace:
            raise ValueError("Existing output requires --resume with identical inputs/code/runtime")
    elif (output / "generations.jsonl").exists() or (output / "workers").exists():
        raise ValueError("Missing parallel run manifest; use a fresh output")
    existing, _, previous_gamma = collect(output, tasks, namespace)
    remaining = [key for key in tasks if key not in existing]
    plan = list(make_plan(args, cfg, remaining, output, namespace))
    print(
        f"{len(tasks)} tasks: {len(existing)} saved, {len(remaining)} remaining, {len(plan)} workers",
        flush=True,
    )
    for folder, gpu, shard, cmd in plan:
        print(
            f"{folder.name} GPU={gpu} tasks={shard['expected_answers']} {subprocess.list2cmdline(cmd)}",
            flush=True,
        )
    if args.dry_run:
        return 0
    base = {**frozen, "run_sha256": namespace, "expected_answers": len(tasks)}
    atomic_json(manifest_path, {**base, "status": "running", "gamma_by_condition": previous_gamma})
    stopped, failed = False, []
    try:
        failed = execute(plan, args)
    except KeyboardInterrupt:
        stopped = True
    rows, workers, gamma = collect(output, tasks, namespace)
    complete = len(rows) == len(tasks)
    if complete:
        ordered = [rows[key] for key in tasks]
        if len({row["answer_id"] for row in ordered}) != len(tasks):
            raise ValueError("Merged answer IDs are not unique")
        atomic_rows(output / "generations.jsonl", ordered)
        export_judge(output, ordered, namespace)
    atomic_json(
        manifest_path,
        {
            **base,
            "status": "complete" if complete else "incomplete",
            "completed": len(rows),
            "failed_workers": failed,
            "interrupted": stopped,
            "gamma_by_condition": gamma,
            "workers": {
                name: {
                    "run_sha256": m["run_sha256"],
                    "runtime": m["runtime"],
                    "status": m["status"],
                }
                for name, m in workers.items()
            },
        },
    )
    print(
        f"{len(rows)}/{len(tasks)} answers {'merged' if complete else 'saved in workers'}",
        flush=True,
    )
    return 130 if stopped else int(not complete or bool(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, choices=("caa24", "hys"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=4,
        help="H200 141GB: 4–5; A100/H100 80GB: 2–3; 40/48GB: 1.",
    )
    parser.add_argument("--stagger-sec", type=float, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gains", "--selected-caa-config", type=Path)
    parser.add_argument(
        "--mock", action="store_true", help="Run the supplied job with CPU mock weights only"
    )
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if (
        any(not g.isascii() or not g.isdecimal() or str(int(g)) != g for g in gpus)
        or len(set(gpus)) != len(gpus)
        or args.workers_per_gpu < 1
        or not 0 <= args.stagger_sec < float("inf")
        or (not args.model and not args.mock)
    ):
        parser.error("Invalid GPU list, worker count, stagger or missing --model")
    cfg = read(args.job)
    methods = {c["method"] for c in cfg["conditions"]}
    if "tasks" in cfg or args.runner not in methods:
        parser.error("--runner must match this job; pass whole jobs, not worker shards")
    prompts = read_rows(PACKAGE / "banks" / f"{cfg['bank']}.jsonl")
    job = resolve_job(cfg, prompts, read(args.gains) if args.gains else None)
    tasks = scheduled_tasks(job, prompts)
    # Freeze effective CAA parameters into each shard before any worker starts;
    # the editable selection file need not remain unchanged during the run.
    cfg = {
        **job,
        "conditions": [
            c
            if c["method"] == "caa24"
            else {k: v for k, v in c.items() if k not in ("alpha", "beta")}
            for c in job["conditions"]
        ],
    }
    source = read(PACKAGE / "directions/manifest.json")
    runtime = (
        {"kind": "cpu-mock", "model": "synthetic"}
        if args.mock
        else {"kind": "single-native", **model_provenance(args.model, inspect_device=False)}
    )
    directions = {
        m: "synthetic mock tensors" if args.mock else source["methods"][m] for m in methods
    }
    frozen = identity(job, prompts, directions, runtime, source["system_prompt"])
    frozen["code_sha256"]["jobs/run_parallel.py"] = sha(Path(__file__))
    namespace = fingerprint_of(frozen)
    if args.dry_run:
        return launch(args, cfg, tasks, frozen, namespace)
    with output_lock(args.output.resolve()):
        return launch(args, cfg, tasks, frozen, namespace)


if __name__ == "__main__":
    sys.exit(main())
