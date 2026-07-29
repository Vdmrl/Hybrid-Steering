#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/student4/Hybrid-Steering-vladimir}"
PYTHON="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/outputs/composition-generation-queue}"
RUNNER="$ROOT/experiments/composition-generation-queue/run.py"
DATA="$ROOT/experiments/composition-data"
PROMPTS="$DATA/test/accepted.jsonl"
CANDOR_PAIRS="${CANDOR_PAIRS:-/home/student4/hybrid-steering/artifacts/candor_filter/factual_v1/accepted.jsonl}"
BASE_DIRECTIONS="${BASE_DIRECTIONS:-/home/student4/hybrid-steering/outputs/composition_base_9b_356_v1/directions}"
OPTIMISM_OUTPUT="${OPTIMISM_OUTPUT:-$ROOT/outputs/optimism-factorial-extension}"
OPTIMISM_PAIRS="${OPTIMISM_PAIRS:-$ROOT/experiments/optimism-factorial-extension/data/optimism_pairs.jsonl}"
JOY_PAIRS="${JOY_PAIRS:-$ROOT/experiments/composition-generation-queue/data/joy_pairs.jsonl}"

mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/queue.lock"
flock -n 9 || { echo "queue already running"; exit 1; }
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

common=(
  --base-directions-dir "$BASE_DIRECTIONS"
  --optimism-direction "$OPTIMISM_OUTPUT/directions/optimism.safetensors"
  --candor-pairs "$CANDOR_PAIRS"
  --data-dir "$DATA"
  --optimism-pairs "$OPTIMISM_PAIRS"
  --joy-pairs "$JOY_PAIRS"
  --prompts "$PROMPTS"
  --output-dir "$OUTPUT"
  --max-new-tokens 256
)

retry() {
  local phase="$1"
  local attempt
  for attempt in 1 2 3; do
    echo "[$(date -Is)] phase=$phase attempt=$attempt"
    if "$PYTHON" "$RUNNER" "$phase" "${common[@]}"; then
      touch "$OUTPUT/$phase.DONE"
      return 0
    fi
    echo "[$(date -Is)] phase=$phase failed; retrying"
  done
  return 1
}

"$PYTHON" "$ROOT/experiments/composition-generation-queue/prepare_joy.py" \
  --output "$JOY_PAIRS" --count 128
retry svd
retry activation
retry joy
retry norm
touch "$OUTPUT/ALL.DONE"
echo "[$(date -Is)] all phases complete"
