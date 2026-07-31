#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/student4/Hybrid-Steering-exp5}"
OUTPUT="${OUTPUT:-/home/student4/Hybrid-Steering-exp5-output}"
PYTHON="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
export PYTHONPATH="$ROOT/steering/src:$ROOT/judge/src${PYTHONPATH:+:$PYTHONPATH}"

while [[ ! -f "$OUTPUT/main.DONE" ]]; do sleep 60; done
[[ -f "$OUTPUT/extension.DONE" ]] && exit 0

CUDA_VISIBLE_DEVICES=3 "$PYTHON" \
  "$ROOT/experiments/five-concept-clamp/run.py" extension \
  --directions-dir "$OUTPUT/source-directions" \
  --first-person-pairs "$OUTPUT/data/first-person.jsonl" \
  --dev-prompts "$OUTPUT/data/dev.jsonl" \
  --test-prompts "$OUTPUT/data/test.jsonl" \
  --output-dir "$OUTPUT"
touch "$OUTPUT/extension.DONE"
