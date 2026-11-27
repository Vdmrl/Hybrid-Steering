#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
: "${MODEL:?Set MODEL to the local pinned BF16 Qwen3.5-9B snapshot}"
GPUS=${GPUS:-0}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
flags=(--resume)
for arg in "$@"; do
  case "$arg" in
    --resume) ;;
    --dry-run) flags+=(--dry-run) ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[[ $GPUS =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || { echo 'Invalid GPUS list' >&2; exit 2; }
IFS=, read -ra devices <<< "$GPUS"
declare -A seen=()
for gpu in "${devices[@]}"; do
  [[ ! ${seen[$gpu]+present} ]] || { echo "Duplicate GPU: $gpu" >&2; exit 2; }
  seen[$gpu]=1
done

children=()
cleanup() {
  if (("${#children[@]}")); then
    kill -TERM "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run() {
  exec "${PYTHON_BIN:-python}" -B jobs/run_parallel.py --runner "$1" --model "$MODEL" \
    --job "jobs/$2.json" --output "outputs/$3" --gpus "$4" \
    --workers-per-gpu "$WORKERS_PER_GPU" --stagger-sec "${STAGGER_SEC:-12}" "${flags[@]}"
}

# Track the currently live launcher in each lane so a signal can be
# forwarded through the queue without leaving its GPU workers behind.
one() {
  local status=0
  (run "$@") &
  children=("$!")
  wait "${children[0]}" || status=$?
  children=()
  return "$status"
}

caa_queue() {
  local gpus=$1
  one caa24 job1b_caa_val_independent16 job1b "$gpus"
  one caa24 job4b_caa_best_multilayer_simple50 job4b "$gpus"
  one caa24 job3_caa_best_multilayer_heldout32 job3_best_multilayer "$gpus"
  one caa24 job3b_caa_gdn24_ctrl_heldout32 job3b_gdn24_ctrl "$gpus"
  one caa24 job1c_gdn24_micro_selection8 job1c_gdn24_micro "$gpus"
}

if [[ $GPUS != *,* ]]; then
  one hys job2b_hys_full_singlegpu_parity job2b "$GPUS"
  caa_queue "$GPUS"
else
  (run hys job2b_hys_full_singlegpu_parity job2b "${GPUS%%,*}") &
  hys=$!
  (
    children=()
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    caa_queue "${GPUS#*,}"
  ) &
  caa=$!
  children=("$hys" "$caa")
  status=0
  wait "$hys" || status=$?
  children=("$caa")
  wait "$caa" || status=$?
  children=()
  exit "$status"
fi
