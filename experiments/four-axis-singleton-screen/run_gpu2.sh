#!/usr/bin/env bash
set -euo pipefail

repo=/home/student4/Hybrid-Steering-four-axis-singleton-screen
out=/home/student4/Hybrid-Steering-four-axis-singleton-screen-output
python=/home/student4/hybrid-steering/.venv/bin/python
prompts=/home/student4/Hybrid-Steering-final-feature-screen-output/data/final-holdout-prompts.jsonl
screen="$repo/experiments/four-axis-singleton-screen/run.py"

export CUDA_VISIBLE_DEVICES=2
cd "$repo"

"$python" "$screen" screen --output "$out" --feature technical --rank rank4 \
  --alpha 2 --direction /home/student4/Hybrid-Steering-final-feature-screen-output/directions/technical-rank4.safetensors \
  --prompts "$prompts" --limit 8 --tag screen
"$python" "$screen" screen --output "$out" --feature numbered_list --rank rank4 \
  --alpha 4 --direction /home/student4/Hybrid-Steering-final-feature-screen-output/directions/numbered_list-rank4.safetensors \
  --prompts "$prompts" --limit 8 --tag screen
for spec in "rank4 .5" "full .5" "rank4 1"; do
  read -r rank alpha <<<"$spec"
  "$python" "$screen" screen --output "$out" --feature russian_language --rank "$rank" \
    --alpha "$alpha" --direction "$out/directions/russian_language-$rank.safetensors" \
    --prompts "$prompts" --limit 4 --tag screen
done
