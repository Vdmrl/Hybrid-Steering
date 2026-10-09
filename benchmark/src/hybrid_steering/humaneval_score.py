"""Score saved HumanEval functions in a restricted bubblewrap sandbox."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

from hybrid_steering import language_benchmark as bench
from hybrid_steering.humaneval_benchmark import cases, folder, sha256, tasks


EXECUTOR = Path(__file__).with_name("humaneval_execute_one.py")


def check_sandbox() -> None:
    probe = subprocess.run(
        ["bwrap", "--unshare-user", "--unshare-net", "--unshare-pid", "--uid", "65534",
         "--gid", "65534", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
         "--ro-bind", "/lib64", "/lib64", "--dev", "/dev", "--proc", "/proc",
         "--tmpfs", "/tmp", "/usr/bin/python3", "-c", "print(123)"],
        capture_output=True, text=True, timeout=10,
    )
    if probe.returncode or probe.stdout.strip() != "123":
        raise RuntimeError(f"Bubblewrap isolation unavailable: {probe.stderr[:400]}")


def execute(problem: dict, completion: str, timeout: int = 10) -> dict:
    payload = {"prompt": problem["prompt"], "completion": completion,
               "test": problem["test"], "entry_point": problem["entry_point"]}
    command = ["bwrap", "--unshare-user", "--unshare-net", "--unshare-pid",
               "--unshare-ipc", "--die-with-parent", "--uid", "65534", "--gid", "65534",
               "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
               "--ro-bind", "/lib64", "/lib64", "--dev", "/dev", "--proc", "/proc",
               "--tmpfs", "/tmp", "--dir", "/work", "--chdir", "/tmp",
               "--ro-bind", str(EXECUTOR), "/work/execute_one.py",
               "/usr/bin/python3", "/work/execute_one.py"]
    try:
        process = subprocess.run(command, input=json.dumps(payload), capture_output=True,
                                 text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout"}
    if process.returncode:
        return {"passed": False, "error": f"SandboxExit{process.returncode}"}
    try:
        return json.loads(process.stdout.strip())
    except json.JSONDecodeError:
        return {"passed": False, "error": "InvalidSandboxOutput"}


def score(config_path: Path, data_root: Path) -> None:
    config = json.loads(config_path.read_text())
    dataset_path = data_root / config["dataset"]
    if sha256(dataset_path) != config["dataset_sha256"]:
        raise ValueError("Dataset SHA256 mismatch")
    with gzip.open(dataset_path, "rt") as file:
        problems = {row["task_id"]: row for row in map(json.loads, file)}
    task_rows = tasks(dataset_path)
    prompts = {row["task_id"]: row["prompt"] for row in task_rows}
    check_sandbox()
    for method, layer, strength in cases(config):
        output = folder(data_root, config, method)
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["method"] != method or manifest["protocol"]["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError(f"Incompatible manifest: {output}")
        answers = [row for row in bench.read_jsonl(output / "answers.jsonl")
                   if row["layer"] == layer and float(row["strength"]) == strength]
        if len(answers) != 164 or {row["task_id"] for row in answers} != set(problems):
            raise ValueError(f"Incomplete answers for {method} L{layer} c={strength}")
        if any(row["prompt"] != prompts[row["task_id"]] for row in answers):
            raise ValueError("Saved prompt differs from pinned protocol")
        score_path = output / "scores.jsonl"
        old = bench.read_jsonl(score_path)
        saved = {(row["layer"], float(row["strength"]), row["task_id"]): row for row in old}
        if len(saved) != len(old):
            raise ValueError("Duplicate score rows")
        for row in answers:
            key = (layer, strength, row["task_id"])
            digest = hashlib.sha256(row["completion"].encode()).hexdigest()
            if key in saved:
                if saved[key]["completion_sha256"] != digest:
                    raise ValueError(f"Saved answer changed: {key}")
                continue
            verdict = execute(problems[row["task_id"]], row["completion"])
            result = {"task_id": row["task_id"], "method": method, "layer": layer,
                      "strength": strength, "completion_sha256": digest, **verdict}
            bench.append_jsonl(score_path, [result])
            saved[key] = result
        selected = [saved[(layer, strength, task_id)] for task_id in problems]
        passed = sum(row["passed"] for row in selected)
        metrics_path = output / "metrics.csv"
        previous = list(csv.DictReader(metrics_path.open())) if metrics_path.exists() else []
        previous = [row for row in previous if (int(row["layer"]), float(row["strength"])) != (layer, strength)]
        previous.append({"method": method, "layer": layer, "strength": strength,
                         "n": len(selected), "passed": passed, "pass_at_1": passed / len(selected)})
        with metrics_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["method", "layer", "strength", "n", "passed", "pass_at_1"])
            writer.writeheader()
            writer.writerows(previous)
        print(f"SCORED {method} L{layer} c={strength}: {passed}/164", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/legacy-config/configs/9b/humaneval_russian_9b.json"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ["GDN_DATA_ROOT"]))
    args = parser.parse_args()
    score(args.config, args.data_root)
