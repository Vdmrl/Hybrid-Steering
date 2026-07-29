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
