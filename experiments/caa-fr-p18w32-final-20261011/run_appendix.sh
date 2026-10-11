#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${STEERING_MODEL_PATH:?Set STEERING_MODEL_PATH to the pinned Qwen3.5-9B directory}"
: "${CUDA_VISIBLE_DEVICES:?Set your allocated physical GPU index}"
python -m unittest test_final
python -u native_run.py --config tasks/appendix32.json --output "${STEERING_OUTPUT_DIR:-$PWD/local/appendix}"
