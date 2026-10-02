"""Official IFEval and sandboxed HumanEval scorers for saved pipeline answers."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ifeval(answers: list[dict], dataset: list[dict], evaluator: dict, dataset_path: Path) -> dict:
    root = Path(evaluator["root"]).resolve()
    source = root / "instruction_following_eval/evaluation_lib.py"
    if _sha(source) != evaluator["evaluation_lib_sha256"]:
        raise ValueError("IFEval evaluator hash differs from config")
    if {row["prompt"] for row in answers} != {row["prompt"] for row in dataset}:
        raise ValueError("IFEval answers do not cover the exact prompt set")
    sys.path.insert(0, str(root))
    try:
        from instruction_following_eval import evaluation_lib
    finally:
        sys.path.pop(0)
    # The official evaluator uses a prompt->response map and random calls.
    response = {row["prompt"]: row["response"] for row in answers}
    inputs = evaluation_lib.read_prompt_list(dataset_path)
    random.seed(0)
    strict = [evaluation_lib.test_instruction_following_strict(row, response) for row in inputs]
    random.seed(0)
    loose = [evaluation_lib.test_instruction_following_loose(row, response) for row in inputs]
    return {
        "prompt_strict": sum(row.follow_all_instructions for row in strict) / len(strict),
        "prompt_loose": sum(row.follow_all_instructions for row in loose) / len(loose),
        "instruction_strict": sum(sum(row.follow_instruction_list) for row in strict)
        / sum(len(row.follow_instruction_list) for row in strict),
        "instruction_loose": sum(sum(row.follow_instruction_list) for row in loose)
        / sum(len(row.follow_instruction_list) for row in loose),
    }


def _sandbox_command(executor: Path) -> list[str]:
    return [
        "bwrap",
        "--unshare-user",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--die-with-parent",
        "--uid",
        "65534",
        "--gid",
        "65534",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--chdir",
        "/tmp",
        "--ro-bind",
        str(executor),
        "/work/execute_one.py",
        "/usr/bin/python3",
        "/work/execute_one.py",
    ]


def humaneval(answers: list[dict], dataset: list[dict]) -> dict:
    executor = Path(__file__).with_name("humaneval_execute_one.py").resolve()
    command = _sandbox_command(executor)
    sample = {
        "prompt": "def f():\n",
        "completion": "    return 1\n",
        "test": "def check(f):\n    assert f() == 1\n",
        "entry_point": "f",
    }
    probe = subprocess.run(
        command, input=json.dumps(sample), text=True, capture_output=True, timeout=10
    )
    if probe.returncode or not json.loads(probe.stdout).get("passed"):
        raise RuntimeError("Bubblewrap sandbox unavailable; HumanEval was not scored")
    tasks = {row["task_id"]: row for row in dataset}
    if {row["task_id"] for row in answers} != set(tasks):
        raise ValueError("HumanEval answers do not cover the exact task set")
    verdicts = []
    for row in answers:
        task = tasks[row["task_id"]]
        payload = {key: task[key] for key in ("prompt", "test", "entry_point")}
        payload["completion"] = row["response"]
        try:
            done = subprocess.run(
                command, input=json.dumps(payload), text=True, capture_output=True, timeout=10
            )
        except subprocess.TimeoutExpired:
            verdicts.append(False)
            continue
        if done.returncode:
            verdicts.append(False)
            continue
        verdicts.append(bool(json.loads(done.stdout)["passed"]))
    return {
        "pass_at_1": sum(verdicts) / len(verdicts),
        "passed": sum(verdicts),
        "n": len(verdicts),
        "executor_sha256": _sha(executor),
    }
