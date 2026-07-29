#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="$root/experiments/optimism-factorial-extension"
output="$root/outputs/optimism-factorial-extension"
python_bin="${PYTHON_BIN:-python}"
pairs="$experiment/data/optimism_pairs.jsonl"

retry() {
  local attempt
  for attempt in 1 2 3; do
    "$@" && return 0
    (( attempt == 3 )) && return 1
    sleep 60
  done
}

phase() {
  retry "$python_bin" "$experiment/run.py" "$1" \
    --pairs "$pairs" \
    --prompts "${PROMPTS_FILE:?set PROMPTS_FILE}" \
    --base-directions-dir "${BASE_DIRECTIONS_DIR:?set BASE_DIRECTIONS_DIR}" \
    --base-output-dir "${BASE_OUTPUT_DIR:?set BASE_OUTPUT_DIR}" \
    --french-output "${FRENCH_OUTPUT_DIR:?set FRENCH_OUTPUT_DIR}" \
    --output-dir "$output"
}

mkdir -p "$output"
exec 8>"$output/queue.lock"
flock -n 8 || { echo "optimism factorial queue is already running"; exit 0; }

HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= retry "$python_bin" "$experiment/prepare.py" \
  --output "$pairs"
phase alpha
phase alpha-input
mkdir -p "$output/judge"
retry "$python_bin" -m hybrid_judge.cli \
  "$output/judge-inputs/alpha-optimism.jsonl" \
  "$output/judge/alpha-optimism.jsonl" \
  --config-root "$root/judge" \
  --mode scalar \
  --feature optimism \
  --workers 8
phase select
phase factorial
phase inputs

while IFS=$'\t' read -r rubric input_path output_path; do
  retry "$python_bin" -m hybrid_judge.cli \
    "$input_path" "$output_path" \
    --config-root "$root/judge" \
    --mode pairwise \
    --feature "$rubric" \
    --workers 8
done < "$output/judge-jobs.tsv"

phase language
mkdir -p "$output/language/judge"
for comparison in optimism_single optimism_with_french; do
  retry "$python_bin" -m hybrid_judge.cli \
    "$output/language/judge-inputs/$comparison.jsonl" \
    "$output/language/judge/$comparison.raw.jsonl" \
    --config-root "$root/judge" \
    --mode pairwise \
    --feature optimism \
    --workers 8
done
for comparison in french_single french_with_optimism; do
  retry "$python_bin" -m hybrid_judge.cli \
    "$output/language/judge-inputs/$comparison.jsonl" \
    "$output/language/judge/$comparison.raw.jsonl" \
    --config-root "$root/judge" \
    --mode pairwise \
    --feature french_language \
    --workers 8
done

"$python_bin" "$experiment/summarize.py" \
  --output-dir "$output" \
  --base-output-dir "$BASE_OUTPUT_DIR"
touch "$output/DONE"
