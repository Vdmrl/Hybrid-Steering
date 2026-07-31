#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/student4/Hybrid-Steering-exp4}"
PYTHON="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
OUTPUT="${OUTPUT:-/home/student4/Hybrid-Steering-vladimir/outputs/strong-composition-exp4}"
EXP3_OUTPUT="/home/student4/Hybrid-Steering-vladimir/outputs/composition-normalization-v3"
RUNNER="$ROOT/experiments/strong-composition-exp4/run.py"
DEVICE="${CUDA_VISIBLE_DEVICES:-3}"

export CUDA_VISIBLE_DEVICES="$DEVICE"
export PYTHONPATH="$ROOT/judge/src:$ROOT/steering/src:${PYTHONPATH:-}"
set -a
source /home/student4/Hybrid-Steering-vladimir/.env
set +a

mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/queue.lock"
flock -n 9 || { echo "Experiment #4 queue already running"; exit 1; }

common=(
  --base-directions-dir /home/student4/hybrid-steering/outputs/composition_base_9b_356_v1/directions
  --optimism-direction /home/student4/Hybrid-Steering-vladimir/outputs/optimism-factorial-extension/directions/optimism.safetensors
  --joy-direction /home/student4/Hybrid-Steering-vladimir/outputs/composition-generation-queue/directions/joy.safetensors
  --cached-ranks-dir /home/student4/Hybrid-Steering-vladimir/outputs/composition-generation-queue/directions
  --dev-prompts "$EXP3_OUTPUT/data/dev.jsonl"
  --test-prompts "$EXP3_OUTPUT/data/test.jsonl"
  --baseline-dev-generations "$EXP3_OUTPUT/dev-generations.jsonl"
  --baseline-main-generations "$EXP3_OUTPUT/gdn-generations.jsonl"
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
  if [[ -f "$OUTPUT/$name.DONE" ]]; then
    echo "[$(date -Is)] phase=$name already complete"
    return 0
  fi
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
  local stage="$2"
  local feature
  for feature in joy concrete_language optimism principled_candor answer_quality; do
    local input="$feature"
    [[ "$feature" == "answer_quality" ]] && input="quality"
    retry "judge-$split-$stage-$feature" \
      "$PYTHON" -m hybrid_judge.cli \
      "$OUTPUT/judge/$split/inputs/$input.jsonl" \
      "$OUTPUT/judge/$split/results/$feature.jsonl" \
      --feature "$feature" --workers 8 --config-root "$ROOT/judge"
  done
}

retry self-test "$PYTHON" "$RUNNER" self-test --output-dir "$OUTPUT"
retry dev direct "$PYTHON" "$RUNNER" dev "${common[@]}"
"$PYTHON" "$RUNNER" prepare-dev-judge "${common[@]}"
judge_split dev scale
retry select "$PYTHON" "$RUNNER" select "${common[@]}"

retry all4 direct "$PYTHON" "$RUNNER" all4 "${common[@]}"
"$PYTHON" "$RUNNER" prepare-main-judge "${common[@]}"
judge_split main all4

retry singletons direct "$PYTHON" "$RUNNER" singletons "${common[@]}"
"$PYTHON" "$RUNNER" prepare-main-judge "${common[@]}"
judge_split main singletons

retry pairs direct "$PYTHON" "$RUNNER" pairs "${common[@]}"
"$PYTHON" "$RUNNER" prepare-main-judge "${common[@]}"
judge_split main pairs

retry summarize "$PYTHON" "$RUNNER" summarize "${common[@]}"
touch "$OUTPUT/ALL.DONE"
echo "[$(date -Is)] Experiment #4 complete"
