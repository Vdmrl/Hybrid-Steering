# Composition generation queue

Generation-only ablations for Qwen3.5-9B. Evaluation is intentionally separate.

The queue runs in priority order:

1. GDN SVD composition: per-feature rank-1, a smaller rank-4 check, and
   rank-1/rank-4 truncation after summing all four features.
2. Classical residual activation steering at layers 10, 20, and 30 with
   alpha 0.5, 1, 2, and 4.
3. Joy/sadness GDN steering alone and combined with optimism.
4. GDN composition rescaled to preserve the root-sum-square direction norm.

Every output row has a stable task ID and is appended immediately, so rerunning
the queue resumes completed work. Large generations and tensor artifacts stay
under the ignored `outputs/` directory.

`run_all.sh` uses GPU selection from `CUDA_VISIBLE_DEVICES`; it does not call a
Judge or any paid API.

## Evaluation summary

`summarize_judge.py` combines two compatible 1–5 Judge outputs without paying
to evaluate the same answer twice. Audited JSON results take precedence;
compact one-digit results fill missing `(prompt_id, answer_id, feature)` keys.
It reports every active feature separately and the joint rate at which all
active features score at least 4.

```bash
python experiments/composition-generation-queue/summarize_judge.py \
  --json-results outputs/composition-generation-queue/judge-final/results \
  --compact-results outputs/composition-generation-queue/judge-compact-v1/results \
  --output reports/composition-generation-queue-summary.json
```
