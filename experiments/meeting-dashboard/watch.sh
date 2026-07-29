#!/usr/bin/env bash
set -eu

PYTHON_BIN="${PYTHON_BIN:-python}"
FOUR_AXIS_SUMMARY="${FOUR_AXIS_SUMMARY:-outputs/four-axis-night/summary.json}"
CALM_FRENCH_SUMMARY="${CALM_FRENCH_SUMMARY:-outputs/calm-french-composition/summary.json}"
CANDOR_FRENCH_SUMMARY="${CANDOR_FRENCH_SUMMARY:-outputs/candor-french-composition/summary.json}"
OPTIMISM_SUMMARY="${OPTIMISM_SUMMARY:-outputs/optimism-factorial-extension/summary.json}"
DASHBOARD_OUTPUT="${DASHBOARD_OUTPUT:-outputs/meeting-dashboard/index.html}"

while true; do
  "$PYTHON_BIN" experiments/meeting-dashboard/build.py \
    --four-axis "$FOUR_AXIS_SUMMARY" \
    --calm-french "$CALM_FRENCH_SUMMARY" \
    --candor-french "$CANDOR_FRENCH_SUMMARY" \
    --optimism "$OPTIMISM_SUMMARY" \
    --output "$DASHBOARD_OUTPUT"

  if [[ -f "$CANDOR_FRENCH_SUMMARY" && -f "$OPTIMISM_SUMMARY" ]]; then
    exit 0
  fi
  sleep 60
done
