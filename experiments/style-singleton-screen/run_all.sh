#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUTPUT="${OUTPUT:-/home/student4/Hybrid-Steering-style-screen-output}"
PYTHON="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
RUNNER="$ROOT/experiments/style-singleton-screen/run.py"
export CUDA_VISIBLE_DEVICES="3"
export PYTHONPATH="$ROOT/steering/src:$ROOT/judge/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT"
"$PYTHON" "$RUNNER" self-test --output-dir "$OUTPUT"
"$PYTHON" "$RUNNER" prepare --output-dir "$OUTPUT" --model "Qwen/Qwen3.5-9B" --pairs 32 --eval-prompts 12
"$PYTHON" "$RUNNER" generate --output-dir "$OUTPUT" --model "Qwen/Qwen3.5-9B" --pairs 32 --eval-prompts 12 --alpha 2
"$PYTHON" "$RUNNER" inputs --output-dir "$OUTPUT" --model "Qwen/Qwen3.5-9B"
echo "generation and blind Judge inputs are ready; run judge/run_screen.sh next"
