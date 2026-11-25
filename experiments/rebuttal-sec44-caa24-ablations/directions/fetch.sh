#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

repo=hybrid-steering/hybrid-steering-concepts
caa_rev=aeff2166d7406fcb59de83e7602f1a736a142a32
hys_rev=82be2f9a41bdc480a8cd1cb86a06b6fb7978f3e4
caa_root=caa/qwen3.5-9b/additive-p18w32-20261011/runtime_reference/directions
hys_suffix=qwen3.5-9b/gdn/rank1-clamp-rss-20261009/rank1.safetensors

while read -r tag caa_name hys_name; do
  caa_src=$(hf download "$repo" "$caa_root/$caa_name.safetensors" --repo-type dataset --revision "$caa_rev")
  hys_src=$(hf download "$repo" "concepts/$hys_name/$hys_suffix" --repo-type dataset --revision "$hys_rev")
  install -D -m 0644 "$caa_src" "directions/caa24/$tag.safetensors"
  install -D -m 0644 "$hys_src" "directions/hys/$tag.safetensors"
done <<'DIRECTIONS'
N numbered numbered
F french en-fr
T technical technical
R theistic_framing atheistic_framing-theistic_framing
P probabilistic_framing binary_framing-probabilistic_framing
DIRECTIONS

sha256sum -c directions/SHA256SUMS.txt
