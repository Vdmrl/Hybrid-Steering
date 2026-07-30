# Composition analysis dashboard

Builds a self-contained HTML dashboard from the compact Judge summaries:

```bash
python experiments/composition-dashboard/build.py
```

The result is written to `outputs/meeting-dashboard-2/index.html`. The
dashboard emphasizes paired method comparisons, uncertainty, feature
bottlenecks, rank-1 retention, composition depth, layer/alpha sensitivity,
and held-out confirmation. Re-run the command after `summary.json`,
`comparisons.json`, and `composition-analysis.json` are updated.

When rebuilding the experiment summary, pass
`--activation-holdout activation-confirm.jsonl` so the 96 untouched prompts
remain separate from the 32-prompt layer/alpha sweep.

`composition-analysis.json` is rebuilt from raw Judge outputs with:

```bash
python experiments/composition-generation-queue/analyze_composition.py \
  --json-results <judge-json-results> \
  --compact-results <judge-compact-results> \
  --activation-holdout <activation-confirm.jsonl> \
  --output experiments/composition-generation-queue/results/composition-analysis.json
```
