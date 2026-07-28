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
  until "$@"; do
    echo "failed; retrying in 60 seconds: $*" >&2
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
run_phase extras
run_phase inputs
run_judge_jobs
retry "$python_bin" "$experiment/summarize.py" --output-dir "$output"
touch "$output/DONE"
