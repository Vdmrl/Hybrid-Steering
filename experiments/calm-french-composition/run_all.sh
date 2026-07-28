#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="$root/experiments/calm-french-composition"
output="$root/outputs/calm-french-composition"
python_bin="${PYTHON_BIN:-python}"
parquet="${FEATURE_STORIES_PARQUET:?set FEATURE_STORIES_PARQUET}"

"$python_bin" "$experiment/prepare.py" \
  --parquet "$parquet" \
  --output "$experiment/data/language_pairs.jsonl"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$python_bin" "$experiment/run.py" \
  --language-pairs "$experiment/data/language_pairs.jsonl" \
  --calm-pairs "$root/experiments/composition-night-v1/data/calm/accepted.jsonl" \
  --prompts "$root/experiments/composition-night-v1/data/test/accepted.jsonl" \
  --output-dir "$output"

mkdir -p "$output/judge"
for comparison in calm_single calm_with_french; do
  "$python_bin" -m hybrid_judge.cli \
    "$output/judge-inputs/$comparison.jsonl" \
    "$output/judge/$comparison.raw.jsonl" \
    --config-root "$root/judge" \
    --judge-version v2 \
    --mode pairwise \
    --feature calm_composure
done
for comparison in french_single french_with_calm; do
  "$python_bin" -m hybrid_judge.cli \
    "$output/judge-inputs/$comparison.jsonl" \
    "$output/judge/$comparison.raw.jsonl" \
    --config-root "$root/judge" \
    --judge-version v2 \
    --mode pairwise \
    --feature french_language
done

"$python_bin" "$experiment/summarize.py" --output-dir "$output"
