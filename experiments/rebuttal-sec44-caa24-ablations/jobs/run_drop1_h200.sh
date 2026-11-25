#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
: "${MODEL:?Set MODEL to the local pinned BF16 Qwen3.5-9B snapshot}"
GPUS=${GPUS:-0}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}

run() {
  "${PYTHON_BIN:-python}" -B jobs/run_parallel.py --runner "$1" --model "$MODEL" \
    --job "jobs/$2.json" --output "outputs/$3" --gpus "$GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" --stagger-sec "${STAGGER_SEC:-12}" --resume
}

run caa24 job1_caa_mega_calib_selection8 job1_mega
printf '\nJob 1 mega complete. Return outputs/job1_mega now while Job 2 runs.\n\n'
run hys job2_hys_ablations_and_ctrl job2
run hys job4a_hys_external_simple50 job4a
