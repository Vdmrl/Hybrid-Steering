#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/student4/Hybrid-Steering-vladimir}"
PYTHON="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/outputs/composition-normalization-v3}"
RUNNER="$ROOT/experiments/composition-normalization-v3/run.py"
PREPARE="$ROOT/experiments/composition-normalization-v3/prepare.py"
DEVICE="${CUDA_VISIBLE_DEVICES:-3}"

export CUDA_VISIBLE_DEVICES="$DEVICE"
export PYTHONPATH="$ROOT/judge/src:$ROOT/steering/src:${PYTHONPATH:-}"
set -a
source "$ROOT/.env"
set +a

mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/queue.lock"
flock -n 9 || { echo "Experiment #3 queue already running"; exit 1; }

common=(
  --base-directions-dir /home/student4/hybrid-steering/outputs/composition_base_9b_356_v1/directions
  --optimism-direction "$ROOT/outputs/optimism-factorial-extension/directions/optimism.safetensors"
  --joy-direction "$ROOT/outputs/composition-generation-queue/directions/joy.safetensors"
  --cached-ranks-dir "$ROOT/outputs/composition-generation-queue/directions"
  --activation-directions "$ROOT/outputs/composition-generation-queue/directions/activation.safetensors"
  --candor-pairs /home/student4/hybrid-steering/artifacts/candor_filter/factual_v1/accepted.jsonl
  --concrete-pairs "$ROOT/experiments/composition-data/concrete/accepted.jsonl"
  --optimism-pairs "$ROOT/experiments/optimism-factorial-extension/data/optimism_pairs.jsonl"
  --joy-pairs "$ROOT/experiments/composition-generation-queue/data/joy_pairs.jsonl"
  --dev-prompts "$OUTPUT/data/dev.jsonl"
  --test-prompts "$OUTPUT/data/test.jsonl"
  --output-dir "$OUTPUT"
  --max-new-tokens 256
  --quality-test 32
)

direct() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy "$@"
}

retry() {
  local name="$1"
  shift
  local attempt
  for attempt in 1 2 3; do
    echo "[$(date -Is)] phase=$name attempt=$attempt"
    if "$@"; then
      touch "$OUTPUT/$name.DONE"
      return 0
    fi
  done
  return 1
}

judge_split() {
  local split="$1"
  local feature
  for feature in joy concrete_language optimism principled_candor; do
    retry "judge-$split-$feature" \
      "$PYTHON" -m hybrid_judge.cli \
      "$OUTPUT/judge/$split/inputs/$feature.jsonl" \
      "$OUTPUT/judge/$split/results/$feature.jsonl" \
      --mode trait --feature "$feature" --workers 8 --config-root "$ROOT/judge"
  done
  retry "judge-$split-quality" \
    "$PYTHON" -m hybrid_judge.cli \
    "$OUTPUT/judge/$split/inputs/quality.jsonl" \
    "$OUTPUT/judge/$split/results/quality.jsonl" \
    --mode scalar --feature concrete_language --workers 8 --config-root "$ROOT/judge"
}

direct "$PYTHON" "$PREPARE" --output-dir "$OUTPUT/data" --dev 32 --test 128
retry smoke direct "$PYTHON" "$RUNNER" smoke "${common[@]}"
retry dev direct "$PYTHON" "$RUNNER" dev "${common[@]}"
retry prepare-dev-judge "$PYTHON" "$RUNNER" prepare-dev-judge "${common[@]}"
judge_split dev
retry select "$PYTHON" "$RUNNER" select "${common[@]}"
retry gdn direct "$PYTHON" "$RUNNER" gdn "${common[@]}"
retry activation direct "$PYTHON" "$RUNNER" activation "${common[@]}"
retry prepare-main-judge "$PYTHON" "$RUNNER" prepare-main-judge "${common[@]}"
judge_split main
retry summarize "$PYTHON" "$RUNNER" summarize "${common[@]}"
touch "$OUTPUT/ALL.DONE"
echo "[$(date -Is)] Experiment #3 complete"
