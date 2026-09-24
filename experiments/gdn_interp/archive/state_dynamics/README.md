# GDN state dynamics

Prepare matched coherent, sentence-shuffled, and repeated PG-19 sequences,
extract per-head state metrics, then build interactive dashboards.

```bash
RUN=artifacts/state_dynamics/$(date -u +%Y%m%d-%H%M%S)
mkdir -p "$RUN/logs"

uv run python experiments/state_dynamics/prepare_sequences.py --run-dir "$RUN"
CUDA_VISIBLE_DEVICES=0 uv run python experiments/state_dynamics/extract.py \
  --run-dir "$RUN" --svd-driver gesvda \
  > "$RUN/logs/extract.log" 2>&1
uv run python experiments/state_dynamics/plot.py --run-dir "$RUN"
```

For a smoke run, add `--count 3 --length 256` during preparation and
`--layers 0 12 23 --heads 0 15 31` during extraction.
