# Dashboard 5 — five-concept clamp

This experiment dashboard is a read-only analysis of the completed
`five-concept-clamp` run. It never calls the GPU or the Judge API.

## Build from a compact summary

```powershell
python experiments/meeting-dashboard-5/build.py `
  --summary experiments/meeting-dashboard-5/exp5_summary.json
```

The page is written to `outputs/meeting-dashboard-5/index.html`. The outputs
directory is ignored by Git.

## Rebuild the compact summary

`build_summary.py` consumes the six completed Judge JSONL files and optional
generation JSONL files. It computes paired 95% bootstrap intervals with 10,000
resamples and a fixed seed, keeps integer trait scores as the primary endpoint,
and stores probability-weighted expected scores as a secondary endpoint.

```powershell
python experiments/meeting-dashboard-5/build_summary.py `
  --results-dir <judge-results> `
  --generations-dir <generation-root> `
  --selection <selection.json> `
  --output experiments/meeting-dashboard-5/exp5_summary.json
```

Only the compact summary, builder, tests, and this README belong in Git. Raw
generations, raw Judge JSONL, and model artifacts must remain outside the
repository.

The page distinguishes trait `n` from quality `n`, uses percentage points only
for rate deltas, reports missing blocks explicitly, and does not claim pairwise
rank-4 contrasts when Exp5 did not generate them.
