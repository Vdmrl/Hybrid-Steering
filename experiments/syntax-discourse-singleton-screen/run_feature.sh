#!/usr/bin/env bash
set -euo pipefail

feature="$1"
gpu="$2"
repo="${REPO:-/home/student4/Hybrid-Steering-syntax-screen}"
out="${OUT:-/home/student4/Hybrid-Steering-syntax-screen-output}/${feature}"
python="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
source_data="${SOURCE_DATA:-/home/student4/Hybrid-Steering-style-screen-output/sources/synthetic-text-transformation-dataset.parquet}"
prompts="${PROMPTS:-/home/student4/Hybrid-Steering-final-feature-screen-output/data/final-holdout-prompts.jsonl}"
runner="$repo/experiments/syntax-discourse-singleton-screen/run.py"

export CUDA_VISIBLE_DEVICES="$gpu"
"$python" "$runner" prepare --feature "$feature" --output "$out" --source "$source_data"
"$python" "$runner" build-direction --feature "$feature" --output "$out"
"$python" "$runner" screen --feature "$feature" --output "$out" --prompts "$prompts" \
  --spec rank1:2 --spec rank1:4 --spec rank4:2 --spec rank4:4 --limit 4
