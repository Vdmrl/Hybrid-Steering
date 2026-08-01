#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/student4/Hybrid-Steering-urgent4}"
OUTPUT="${OUTPUT:-/home/student4/Hybrid-Steering-urgent4-output}"
PYTHON="${PYTHON:-/home/student4/hybrid-steering/.venv/bin/python}"
RUNNER="$ROOT/experiments/urgent-russian-refusal/run.py"
export PYTHONPATH="$ROOT/steering/src:$ROOT/judge/src${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(
  --directions-dir "$OUTPUT/source-directions"
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
ln -sfn /home/student3/GDN/F4_ru_en/data/deltaS_ruen_clean.npz "$OUTPUT/source-directions/russian_language.npz"
ln -sfn /home/student4/Hybrid-Steering-exp5-bullets-output/source-directions/optimism.safetensors "$OUTPUT/source-directions/optimism.safetensors"
ln -sfn /home/student3/GDN/F3_casual_formal/data/deltaS_casualformal.npz "$OUTPUT/source-directions/casualness.npz"
ln -sfn /home/student3/GDN/F2_refusal_scale/data/deltaS_refusal.npz "$OUTPUT/source-directions/refusal.npz"
ln -sfn /home/student4/Hybrid-Steering-exp5-bullets-output/data/dev.jsonl "$OUTPUT/data/dev.jsonl"
ln -sfn /home/student4/Hybrid-Steering-exp5-bullets-output/data/test.jsonl "$OUTPUT/data/test.jsonl"
ln -sfn /home/student1/shared/english_questions.jsonl "$OUTPUT/data/factual.jsonl"
retry self-test "$PYTHON" "$RUNNER" self-test "${COMMON[@]}"
retry smoke env CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$RUNNER" smoke "${COMMON[@]}"
retry main env CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$RUNNER" main "${COMMON[@]}"
retry factuality env CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$RUNNER" factuality "${COMMON[@]}" --factual-prompts "$OUTPUT/data/factual.jsonl"
retry judge-input "$PYTHON" "$ROOT/experiments/urgent-russian-refusal/prepare_judge.py" "$OUTPUT/main-generations.jsonl" "$OUTPUT/judge-input.jsonl"
for feature in optimism casualness answer_quality; do
  retry "judge-$feature" "$PYTHON" -m hybrid_judge.cli \
    "$OUTPUT/judge-input.jsonl" "$OUTPUT/judge/$feature.jsonl" \
    --feature "$feature" --workers 8 --config-root "$ROOT/judge"
done
touch "$OUTPUT/ALL.DONE"
echo "[$(date -Is)] Urgent four-axis generation complete"
