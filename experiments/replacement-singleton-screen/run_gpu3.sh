#!/usr/bin/env bash
set -euo pipefail

repo=/home/student4/Hybrid-Steering-replacement-singleton-screen
out=/home/student4/Hybrid-Steering-replacement-singleton-screen-output
python=/home/student4/hybrid-steering/.venv/bin/python
run="$repo/experiments/replacement-singleton-screen/run.py"
prompts=/home/student4/Hybrid-Steering-final-feature-screen-output/data/final-holdout-prompts.jsonl

export CUDA_VISIBLE_DEVICES=3
cd "$repo"
"$python" "$run" prepare-past --output "$out" \
  --style-root /home/student4/Hybrid-Steering-style-screen-output/sources/StylePTB
"$python" "$run" build-direction --output "$out" --feature past_tense --batch 8
"$python" "$run" screen --output "$out" --feature past_tense --prompts "$prompts" \
  --spec full:2 --spec rank4:2 --spec rank1:2 --spec rank4:4 --limit 4
"$python" "$run" screen --output "$out" --feature persuasive --prompts "$prompts" \
  --direction /home/student4/Hybrid-Steering-final-feature-screen-output/directions/persuasive-rank1.safetensors \
  --spec rank1:4 --spec rank1:8 --limit 8
