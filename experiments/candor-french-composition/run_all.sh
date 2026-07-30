#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="$root/experiments/candor-french-composition"
output="$root/outputs/candor-french-composition"
python_bin="${PYTHON_BIN:-python}"

retry() {
  local attempt
  for attempt in 1 2 3; do
    "$@" && return 0
    (( attempt == 3 )) && return 1
    sleep 60
  done
}

mkdir -p "$output"
exec 8>"$output/queue.lock"
flock -n 8 || { echo "candor/French queue is already running"; exit 0; }

"$python_bin" "$experiment/run.py" \
  --french-output "${FRENCH_OUTPUT_DIR:?set FRENCH_OUTPUT_DIR}" \
  --candor-direction "${CANDOR_DIRECTION:?set CANDOR_DIRECTION}" \
  --prompts "${PROMPTS_FILE:?set PROMPTS_FILE}" \
  --output-dir "$output"

mkdir -p "$output/judge"
for comparison in candor_single candor_with_french; do
  retry "$python_bin" -m hybrid_judge.cli \
    "$output/judge-inputs/$comparison.jsonl" \
    "$output/judge/$comparison.raw.jsonl" \
    --config-root "$root/judge" \
    --mode pairwise \
    --feature principled_candor \
    --workers 8
done
for comparison in french_single french_with_candor; do
  retry "$python_bin" -m hybrid_judge.cli \
    "$output/judge-inputs/$comparison.jsonl" \
    "$output/judge/$comparison.raw.jsonl" \
    --config-root "$root/judge" \
    --mode pairwise \
    --feature french_language \
    --workers 8
done

"$python_bin" "$experiment/summarize.py" --output-dir "$output"
touch "$output/DONE"
