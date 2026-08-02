#!/usr/bin/env bash
set -euo pipefail

repo=/home/student4/Hybrid-Steering-replacement-singleton-screen
out=/home/student4/Hybrid-Steering-confidence-mini-output
python=/home/student4/hybrid-steering/.venv/bin/python
run="$repo/experiments/replacement-singleton-screen/run.py"
prompts=/home/student4/Hybrid-Steering-final-feature-screen-output/data/final-holdout-prompts.jsonl

export CUDA_VISIBLE_DEVICES=2
cd "$repo"
"$python" "$run" repair-confidence --output "$out" --count 10 --candidates 24 \
  --batch 8 --source-pairs /home/student4/Hybrid-Steering-replacement-singleton-screen-output/data/confident_pairs.json
"$python" "$run" build-direction --output "$out" --feature confident --batch 8
"$python" "$run" screen --output "$out" --feature confident --prompts "$prompts" \
  --spec rank1:4 --spec rank4:4 --limit 4 --tag minimal
