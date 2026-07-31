# Experiment #4: strong GDN composition

This experiment reuses the four Exp3 directions and its fixed 32/128 prompt
split. Exp3 matched singleton strength between GDN and activation; Exp4 instead
uses each GDN direction's strongest quality-safe dev alpha.

The experiment fills the missing rank-by-normalization cell:

| | Raw addition | RSS |
|---|---|---|
| Per-feature rank 1 | yes | yes |
| Per-feature rank 4 | yes | yes |

The autonomous queue runs all-four first, then singletons, then all six pairs.
Each completed block is judged before the next block starts, so the primary
result survives an interrupted queue.

```bash
nohup bash experiments/strong-composition-exp4/run_all.sh \
  > /home/student4/Hybrid-Steering-vladimir/outputs/strong-composition-exp4/queue.log \
  2>&1 &
```

The queue is resumable by stable prompt and answer IDs. Large generations and
Judge artifacts remain outside Git.
