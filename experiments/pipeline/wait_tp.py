"""Wait for a specified GPU pair, then run a prepared TP command once."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path


def gpu_status() -> dict[int, tuple[str, int, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = {}
    for line in output.splitlines():
        index, uuid, total, used = (part.strip() for part in line.split(","))
        result[int(index)] = (uuid, int(total), int(used))
    return result


def free_pair(
    state: dict[int, tuple[str, int, int]], ids: tuple[int, int], *, max_used: int = 700
) -> tuple[str, str] | None:
    if any(index not in state or state[index][1] < 15000 for index in ids):
        raise ValueError(f"expected two A4000 GPUs at physical indices {ids}")
    if any(state[index][2] > max_used for index in ids):
        return None
    return state[ids[0]][0], state[ids[1]][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=(1, 2))
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or len(set(args.gpus)) != 2:
        parser.error("need two distinct GPUs and a positive poll interval")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command after --")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_file = root / "queue-status.json"
    locks = root.parents[1] / "gpu-locks"
    locks.mkdir(parents=True, exist_ok=True)
    ids = tuple(args.gpus)
    while True:
        state = {
            "state": "waiting",
            "time": time.time(),
            "poll_seconds": args.poll_seconds,
            "gpus": ids,
            "inputs_ready": args.ready.is_file(),
        }
        try:
            pair = free_pair(gpu_status(), ids) if state["inputs_ready"] else None
            if pair:
                with ExitStack() as stack:
                    try:
                        for index in ids:
                            handle = stack.enter_context((locks / f"gpu-{index}.lock").open("a"))
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pair = None
                    if pair and free_pair(gpu_status(), ids) == pair:
                        env = {
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": ",".join(pair),
                            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        }
                        state.update(state="running", gpu_uuids=pair, command=command)
                        status_file.write_text(json.dumps(state, indent=2) + "\n")
                        code = subprocess.call(command, env=env)
                        state.update(
                            state="complete" if code == 0 else "failed",
                            exit_code=code,
                            time=time.time(),
                        )
                        status_file.write_text(json.dumps(state, indent=2) + "\n")
                        raise SystemExit(code)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            state["error"] = str(error)
        status_file.write_text(json.dumps(state, indent=2) + "\n")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
