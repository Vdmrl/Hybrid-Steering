#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/student4/Hybrid-Steering-exp5}"
OUTPUT="${OUTPUT:-/home/student4/Hybrid-Steering-exp5-output}"
PYTHON="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
RUNNER="$ROOT/experiments/five-concept-clamp/run.py"
export PYTHONPATH="$ROOT/steering/src:$ROOT/judge/src${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(
  --directions-dir "$OUTPUT/source-directions"
  --first-person-pairs "$OUTPUT/data/first-person.jsonl"
  --dev-prompts "$OUTPUT/data/dev.jsonl"
  --test-prompts "$OUTPUT/data/test.jsonl"
  --output-dir "$OUTPUT"
)

retry() {
  local name="$1"; shift
  [[ -f "$OUTPUT/$name.DONE" ]] && return 0
  for attempt in 1 2 3; do
    echo "[$(date -Is)] phase=$name attempt=$attempt"
    if "$@"; then
      touch "$OUTPUT/$name.DONE"
      return 0
    fi
  done
  return 1
}

mkdir -p "$OUTPUT/data" "$OUTPUT/source-directions"
retry self-test "$PYTHON" "$RUNNER" self-test --output-dir "$OUTPUT"
retry first-person-pairs "$PYTHON" "$ROOT/experiments/five-concept-clamp/prepare_first_person.py" "$OUTPUT/data/first-person.jsonl"
retry direction env CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$RUNNER" direction "${COMMON[@]}"
retry smoke env CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$RUNNER" smoke "${COMMON[@]}"
# Judge-based dev selection remains a registered extension. The primary queue
# starts with the conservative Exp4-safe scale and middle soft-clamp beta.
if [[ ! -f "$OUTPUT/selection.json" ]]; then
  printf '{"scale": 0.5, "beta": 0.5, "status": "conservative_preselection"}\n' > "$OUTPUT/selection.json"
fi

retry main env CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$RUNNER" main "${COMMON[@]}"
retry extension env CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$RUNNER" extension "${COMMON[@]}"
touch "$OUTPUT/ALL.DONE"
echo "[$(date -Is)] Experiment #5 generation complete"
