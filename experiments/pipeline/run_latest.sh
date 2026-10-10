#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv sync --extra dev --extra benchmarks
uv run python -c 'import nltk; assert all(nltk.download(name, quiet=True) for name in ("punkt_tab", "punkt"))'
exec uv run python experiments/pipeline/language.py --run-judge "$@"
