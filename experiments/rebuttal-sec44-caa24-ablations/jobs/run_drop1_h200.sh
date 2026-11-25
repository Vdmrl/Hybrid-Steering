#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
: "${MODEL:?Set MODEL to the local pinned BF16 Qwen3.5-9B snapshot}"
GPUS=${GPUS:-0}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
[[ $GPUS =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || { echo 'Invalid GPUS list' >&2; exit 2; }
IFS=, read -ra devices <<< "$GPUS"
declare -A seen=()
for gpu in "${devices[@]}"; do
  [[ ! ${seen[$gpu]+present} ]] || { echo "Duplicate GPU: $gpu" >&2; exit 2; }
  seen[$gpu]=1
done

run() {
  local method=$1 job=$2 name=$3 gpus=$4
  exec "${PYTHON_BIN:-python}" -B jobs/run_parallel.py --runner "$method" --model "$MODEL" \
    --job "jobs/$job.json" --output "outputs/$name" --gpus "$gpus" \
    --workers-per-gpu "$WORKERS_PER_GPU" --stagger-sec "${STAGGER_SEC:-12}" --resume
}

announce() {
  printf '\nJob 1 mega complete. Return outputs/job1_mega now; do not wait for the other jobs.\n\n'
}

# Use subshells so each tracked PID becomes the Python launcher via exec.
# The exit trap forwards interruption and reaps any launchers still running.
pids=()
cleanup() {
  if (("${#pids[@]}")); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

first_status=0
second_status=0
(run caa24 job1_caa_mega_calib_selection8 job1_mega "${GPUS%%,*}") &
first=$!
pids+=("$first")
if [[ $GPUS == *,* ]]; then
  (run hys job2_hys_ablations_and_ctrl job2 "${GPUS#*,}") &
  second=$!
  pids+=("$second")
fi
wait "$first" || first_status=$?
pids=()
if [[ $GPUS == *,* ]]; then pids=("$second"); fi
if ((first_status == 0)); then announce; fi
if [[ $GPUS == *,* ]]; then
  wait "$second" || second_status=$?
elif ((first_status == 0)); then
  (run hys job2_hys_ablations_and_ctrl job2 "$GPUS") &
  second=$!
  pids+=("$second")
  wait "$second" || second_status=$?
fi
pids=()
((first_status == 0 && second_status == 0)) || exit 1
run hys job4a_hys_external_simple50 job4a "$GPUS"
