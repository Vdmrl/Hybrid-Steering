#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUTPUT="${OUTPUT:-/home/student4/Hybrid-Steering-final-feature-screen-output}"
PYTHON="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
BULLETS="${BULLETS:-/home/student4/Hybrid-Steering-exp5-bullets-output/data/bullets.jsonl}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="$ROOT/steering/src:$ROOT/judge/src${PYTHONPATH:+:$PYTHONPATH}"
FEATURE="${FEATURE:-humorous}"
RANK="${RANK:-rank4}"
ALPHA="${ALPHA:-2}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
LIMIT="${LIMIT:-}"
TAG="${TAG:-smoke}"
EXTRA=()
[ -n "$PROMPTS_FILE" ] && EXTRA+=(--prompts-file "$PROMPTS_FILE")
[ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")

mkdir -p "$OUTPUT"
"$PYTHON" "$ROOT/experiments/final-feature-screen/prepare_pairs.py" \
  --output-dir "$OUTPUT" --bullet-source "$BULLETS" --pairs 128
"$PYTHON" "$ROOT/experiments/final-feature-screen/screen.py" run \
  --output-dir "$OUTPUT" --model Qwen/Qwen3.5-9B --feature "$FEATURE" \
  --rank "$RANK" --alpha "$ALPHA" --max-new-tokens 128 --tag "$TAG" "${EXTRA[@]}"
"$PYTHON" "$ROOT/experiments/final-feature-screen/screen.py" summary \
  --output-dir "$OUTPUT" --feature "$FEATURE" --rank "$RANK" --alpha "$ALPHA" \
  --tag "$TAG" "${EXTRA[@]}"
