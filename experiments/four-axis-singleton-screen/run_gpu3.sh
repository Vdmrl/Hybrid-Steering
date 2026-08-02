#!/usr/bin/env bash
set -euo pipefail

repo=/home/student4/Hybrid-Steering-four-axis-singleton-screen
out=/home/student4/Hybrid-Steering-four-axis-singleton-screen-output
python=/home/student4/hybrid-steering/.venv/bin/python
prompts=/home/student4/Hybrid-Steering-final-feature-screen-output/data/final-holdout-prompts.jsonl
screen="$repo/experiments/four-axis-singleton-screen/run.py"

export CUDA_VISIBLE_DEVICES=3
cd "$repo"

for rank in full rank4 rank1; do
  "$python" "$screen" screen --output "$out" --feature passive_voice --rank "$rank" \
    --alpha 2 --prompts "$prompts" --limit 4 --tag screen
done
