# Composition analysis dashboard

Builds a self-contained HTML dashboard from the compact Judge summaries:

```bash
python experiments/composition-dashboard/build.py
```

The result is written to `outputs/meeting-dashboard-2/index.html`. The
dashboard emphasizes method comparisons, uncertainty, feature bottlenecks,
composition depth, layer/alpha sensitivity, and held-out confirmation. Re-run
the command after `summary.json` and `comparisons.json` are updated.

When rebuilding the experiment summary, pass
`--activation-holdout activation-confirm.jsonl` so the 96 untouched prompts
remain separate from the 32-prompt layer/alpha sweep.
