# Experiment 4 dashboard

The Russian-language dashboard explains the autonomous Exp4 queue (strong
composition with rank × RSS-normalization ablations) and can render its compact
`summary.json` when the Judge phase finishes. The results view includes
all-four conditions, feature deltas, all six pairs, paired contrasts, singleton
retention, CI-based neutral conclusions, and Judge usage. It never calls the GPU
or Judge API.

```powershell
python experiments/meeting-dashboard-4/build.py
```

Without a summary file it renders the plan and a clear waiting state. Partial
or missing conditions are shown as `missing`, not as zero effects. To embed
completed results:

```powershell
python experiments/meeting-dashboard-4/build.py `
  --summary outputs/strong-composition-exp4/summary.json
```

It writes `outputs/meeting-dashboard-4/index.html`. Large generations and raw
Judge artifacts stay outside Git; only the compact summary is consumed by the
dashboard.

The Exp4 summary records `quality_n` separately from each condition's trait
`n`; older summaries without that field are shown as `missing` rather than
inferred.
