# Chunk-boundary dynamics

The similarity collector samples all GDN layers at token positions 1–31 and
then chunk boundaries through 8192. It writes per-document, layer, and head
metrics plus Plotly dashboards.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.chunk_dynamics.run \
  --output artifacts/chunk_dynamics/similarities --documents 200 --batch-size 32
```

State rank and Frobenius norms are collected together. Set `--max-position` to
shorten a run without changing its dataset or document count.

```bash
RUN=artifacts/chunk_dynamics/ranks_200
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.chunk_dynamics.ranks \
  --output "$RUN" --documents 200 --max-position 1024
uv run python -m experiments.chunk_dynamics.rank_plots "$RUN/metrics.parquet" --output "$RUN"
uv run python -m experiments.chunk_dynamics.norm_plots "$RUN/metrics.parquet" --output "$RUN"
```

Value-similarity controls require two passes. The first writes the three local
controls and accumulates per-head value means and global left-vector covariance;
the second uses those statistics for centered and global-axis controls.

```bash
RUN=artifacts/chunk_dynamics/controls
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.chunk_dynamics.controls first \
  --output "$RUN/first.parquet" --statistics "$RUN/statistics.pt"
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.chunk_dynamics.controls global \
  --output "$RUN/global.parquet" --statistics "$RUN/statistics.pt"
uv run python -m experiments.chunk_dynamics.control_plots \
  "$RUN/first.parquet" "$RUN/global.parquet" --output "$RUN"
```
