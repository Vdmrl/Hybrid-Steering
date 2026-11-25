#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  '') [[ $# -eq 0 ]] || exit 2; source_mode=hf ;;
  --from-dir)
    [[ $# -eq 2 ]] || { echo 'Usage: fetch.sh --from-dir DIRECTORY' >&2; exit 2; }
    source_mode=local
    source_dir=$(cd -- "$2" && pwd -P)
    ;;
  --github-release)
    [[ $# -eq 3 ]] || { echo 'Usage: fetch.sh --github-release OWNER/REPO TAG' >&2; exit 2; }
    source_mode=github
    github_repo=$2
    github_tag=$3
    ;;
  *) echo 'Usage: fetch.sh [--from-dir DIRECTORY | --github-release OWNER/REPO TAG]' >&2; exit 2 ;;
esac
cd -- "$(dirname -- "$0")/.."

repo=hybrid-steering/hybrid-steering-concepts
caa_rev=aeff2166d7406fcb59de83e7602f1a736a142a32
hys_rev=82be2f9a41bdc480a8cd1cb86a06b6fb7978f3e4
caa_root=caa/qwen3.5-9b/additive-p18w32-20261011/runtime_reference/directions
hys_suffix=qwen3.5-9b/gdn/rank1-clamp-rss-20261009/rank1.safetensors

stage=$(mktemp -d)
trap 'rm -r -- "$stage"' EXIT
if [[ $source_mode == github ]]; then
  # gh accepts GH_TOKEN or GITHUB_TOKEN for an existing mirror release.
  gh release download --repo "$github_repo" --dir "$stage" \
    --pattern 'caa24-?.safetensors' --pattern 'hys-?.safetensors' -- "$github_tag"
elif [[ $source_mode == local && -d $source_dir/directions ]]; then
  source_dir=$source_dir/directions
fi

while read -r tag caa_name hys_name; do
  case "$source_mode" in
    hf)
      # The HF dataset is access-controlled; use HF_TOKEN or hf auth login.
      caa_src=$(hf download "$repo" "$caa_root/$caa_name.safetensors" --repo-type dataset --revision "$caa_rev")
      hys_src=$(hf download "$repo" "concepts/$hys_name/$hys_suffix" --repo-type dataset --revision "$hys_rev")
      ;;
    local)
      caa_src=$source_dir/caa24/$tag.safetensors
      hys_src=$source_dir/hys/$tag.safetensors
      ;;
    github)
      caa_src=$stage/caa24-$tag.safetensors
      hys_src=$stage/hys-$tag.safetensors
      ;;
  esac
  install -D -m 0644 "$caa_src" "$stage/directions/caa24/$tag.safetensors"
  install -D -m 0644 "$hys_src" "$stage/directions/hys/$tag.safetensors"
done <<'DIRECTIONS'
N numbered numbered
F french en-fr
T technical technical
R theistic_framing atheistic_framing-theistic_framing
P probabilistic_framing binary_framing-probabilistic_framing
DIRECTIONS

checksums=$PWD/directions/SHA256SUMS.txt
(cd -- "$stage" && sha256sum -c "$checksums")
for method in caa24 hys; do
  for tag in N F T R P; do
    file=directions/$method/$tag.safetensors
    install -D -m 0644 "$stage/$file" "$file"
  done
done
sha256sum -c directions/SHA256SUMS.txt
