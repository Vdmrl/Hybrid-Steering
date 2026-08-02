#!/usr/bin/env bash
set -euo pipefail

repo=/home/student4/Hybrid-Steering-replacement-singleton-screen
out=/home/student4/Hybrid-Steering-replacement-singleton-screen-output
python=/home/student4/hybrid-steering/.venv/bin/python
run="$repo/experiments/replacement-singleton-screen/run.py"
prompts=/home/student4/Hybrid-Steering-final-feature-screen-output/data/final-holdout-prompts.jsonl

export CUDA_VISIBLE_DEVICES=2
cd "$repo"
"$python" "$run" prepare-confidence --output "$out" --batch 8 \
  --source /home/student4/Hybrid-Steering-style-screen-output/sources/synthetic-text-transformation-dataset.parquet
"$python" "$run" build-direction --output "$out" --feature confident --batch 8
"$python" "$run" screen --output "$out" --feature confident --prompts "$prompts" \
  --spec full:2 --spec rank4:2 --spec rank1:2 --spec rank4:4 --limit 4
