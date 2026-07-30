#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="$root/experiments/four-axis-night"
output="$root/outputs/four-axis-night"
python_bin="${PYTHON_BIN:-python}"

if [[ -f "$root/.env" ]]; then
  set -a
  source "$root/.env"
  set +a
fi

directions="${DIRECTIONS_DIR:?set DIRECTIONS_DIR}"
candor_pairs="${CANDOR_PAIRS:?set CANDOR_PAIRS}"

mkdir -p "$output"
exec 9>"$output/queue.lock"
flock -n 9 || { echo "four-axis queue is already running"; exit 0; }

retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    if (( attempt == 3 )); then
      echo "failed after 3 attempts: $*" >&2
      return 1
    fi
    echo "failed; retrying in 60 seconds ($attempt/3): $*" >&2
    sleep 60
  done
}

run_phase() {
  retry "$python_bin" "$experiment/run.py" "$1" \
    --directions-dir "$directions" \
    --data-dir "$root/experiments/composition-data" \
    --candor-pairs "$candor_pairs" \
    --prompts "$root/experiments/composition-data/test/accepted.jsonl" \
    --output-dir "$output" \
    --max-new-tokens 256
}

run_judge_jobs() {
  while IFS=$'\t' read -r feature mode input_path output_path; do
    retry "$python_bin" -m hybrid_judge.cli \
      "$input_path" "$output_path" \
      --config-root "$root/judge" \
      --mode "$mode" \
      --feature "$feature" \
      --workers 8
  done < "$output/judge-jobs.tsv"
}

run_phase alpha
run_phase inputs
run_judge_jobs
run_phase select
run_phase factorial
run_phase inputs
run_judge_jobs
retry "$python_bin" "$experiment/summarize.py" --output-dir "$output"
if [[ "${SKIP_EXTRAS:-0}" != 1 ]]; then
  run_phase extras
  run_phase inputs
  touch "$output/EXTRAS_READY_FOR_JUDGE"
fi
touch "$output/DONE"
retry env CALM_FRENCH_DONORS=62 bash \
  "$root/experiments/calm-french-composition/run_all.sh"
