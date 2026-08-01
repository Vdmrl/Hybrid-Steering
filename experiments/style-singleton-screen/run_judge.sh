#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUTPUT="${OUTPUT:-/home/student4/Hybrid-Steering-style-screen-output}"
PYTHON="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
export PYTHONPATH="$ROOT/judge/src:$ROOT/steering/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$ROOT/experiments/style-singleton-screen/make_judge_config.py" \
  --output-dir "$OUTPUT" --repo-root "$ROOT"
mkdir -p "$OUTPUT/judge-results"
for feature in humorous adjective_emphasis action_emphasis technical persuasive narrative answer_quality; do
  "$PYTHON" -m hybrid_judge.cli \
    "$OUTPUT/judge-inputs/$feature.jsonl" "$OUTPUT/judge-results/$feature.jsonl" \
    --feature "$feature" --workers 8 --config-root "$OUTPUT/judge-config/judge"
done
"$PYTHON" "$ROOT/experiments/style-singleton-screen/compact.py" --output-dir "$OUTPUT"
"$PYTHON" "$ROOT/experiments/style-singleton-screen/run.py" summary \
  --output-dir "$OUTPUT" --model "Qwen/Qwen3.5-9B" --pairs 32 --eval-prompts 12
