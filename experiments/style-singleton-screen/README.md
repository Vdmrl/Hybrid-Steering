# Style singleton screen

This experiment is an exploratory, GPU-only screen of six additional style
directions. It deliberately stops at singleton interventions: no factorial,
composition, clamp, or pair generation is run here.

The runner prepares 32 paired donor examples per feature, extracts the full
GDN direction and a per-layer rank-1 SVD approximation, generates 12 common
neutral prompts at `alpha=2`, and optionally checks `alpha=4` on four prompts
when the first screen is weak but remains readable. Judge inputs are blind to
method, rank, and alpha. New rubrics are written only under the output
directory and are not added to `concepts/features.yaml`.

Run on the server with `CUDA_VISIBLE_DEVICES=3` and `run_all.sh`. Large source,
pair, direction, generation, and Judge artifacts belong in the external
output directory; only this code and the compact `summary.json` are tracked.
