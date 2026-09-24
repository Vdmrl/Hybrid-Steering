# Context-length steering

Extract an optimism-minus-pessimism direction, prepend neutral Wikipedia
prefixes to pessimistic tweets, and generate at raw scales. Defaults use every tweet in
`../shared/twitter_pessimistic_tweets.jsonl`, a tweet-only point (plotted at
its mean input length), exact 128–8192-token contexts, and 256 new response
tokens. Within an example, every Wikipedia-backed length uses the same source,
offset, and complete tweet; shorter fillers are exact token prefixes of longer
ones. Steering sets every GDN layer's `S₀` once before prefill.

```bash
RUN=artifacts/context_length/$(date -u +%Y%m%d-%H%M%S)
mkdir -p "$RUN/logs"

CUDA_VISIBLE_DEVICES=0 uv run python experiments/context_length/extract_direction.py \
  --run-dir "$RUN" --dataset ../shared/optimism_pessimism_stories.jsonl \
  --text-columns concept_text antagonist_text --where language=English \
  --limit 128 --max-length 512 \
  > "$RUN/logs/extract.log" 2>&1

CUDA_VISIBLE_DEVICES=0 uv run python experiments/context_length/prepare_contexts.py \
  --run-dir "$RUN" --sources artifacts/state_dynamics/<run>/inputs/sequences.pt \
  --dataset ../shared/twitter_pessimistic_tweets.jsonl \
  > "$RUN/logs/prepare.log" 2>&1

CUDA_VISIBLE_DEVICES=0 uv run python experiments/context_length/generate.py \
  --run-dir "$RUN" --normalization raw --batch-size 64 --seed 42
```

Judging is intentionally external while the custom judge is developed.
`validation.json` records the exact-length, nested-prefix, and
shared-source checks; `source_metadata.jsonl` records each source and offset.
Raw steering directly scales each head's extracted mean-difference matrix.
`--normalization rms` reproduces legacy runs by applying one global RMS multiplier.
For a real direction (not a smoke test), extract from every paired example
available, e.g. `--limit 2000`, rather than the small subset in the snippet
above.

## Steering timing modes

`generate.py --mode` controls *when* `delta` is added to each GDN layer's
recurrent state, independent of the scale sweep above. Every mode uses one
run directory (its own `contexts.jsonl` / `direction.pt`, or symlinks to a
shared one) and writes `steering_mode` / `period` onto each response row,
so responses from different modes never collide during resume even if they
share a run directory.

- `initial` (default): `S_0 = delta` once before prefill, never touched again
  — the original position-0 experiment.
- `repeated`: `S_0 = delta`, then re-add `delta` after every forward call
  (every generated token). This is `periodic` with `--period 1`.
- `periodic --period N`: `S_0 = delta`, then re-add `delta` every `N` forward
  calls.
- `prompt_end`: no `S_0` preset; add `delta` exactly once, right after the
  prompt is fully processed, before the first generated token.

```bash
for MODE_RUN in every-token periodic-32 periodic-64 periodic-128 prompt-end; do
  RUN="artifacts/context_length/$(date -u +%Y%m%d-%H%M%S)-$MODE_RUN"
  mkdir -p "$RUN/logs"
  cp "$PREVIOUS_RUN/direction.pt" "$RUN/direction.pt"
  cp "$PREVIOUS_RUN/contexts.jsonl" "$RUN/contexts.jsonl"
done

CUDA_VISIBLE_DEVICES=0 uv run python experiments/context_length/generate.py \
  --run-dir "$EVERY_TOKEN_RUN" --normalization raw --mode repeated \
  --scales 21 --batch-size 64 --seed 42

CUDA_VISIBLE_DEVICES=0 uv run python experiments/context_length/generate.py \
  --run-dir "$PERIODIC_32_RUN" --normalization raw --mode periodic --period 32 \
  --scales 21 --batch-size 64 --seed 42
# repeat with --period 64 and --period 128 in their own run dirs

CUDA_VISIBLE_DEVICES=0 uv run python experiments/context_length/generate.py \
  --run-dir "$PROMPT_END_RUN" --normalization raw --mode prompt_end \
  --scales 21 --batch-size 64 --seed 42
```

Apply the custom judge to each run directory when it is available.
Rank-one propagation replay (`collect_propagation.py`) only supports
`--mode initial` runs: its analytic sigma_t/cosine trajectory tracks decay
of a single `S_0` injection and cannot represent re-injected delta events.
For the other three modes, collect state metrics and analyze results with the
custom judge instead.

## Rank-one propagation

Replay the saved scale-1.25 prompt/response tokens through the steered model,
save every token/head metric, average across texts, and build the white HTML:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python experiments/context_length/collect_propagation.py \
  --run-dir "$RUN"
uv run python experiments/context_length/propagation_dashboard.py --run-dir "$RUN"
```

Resume a prepared feature-story run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python experiments/context_length/generate.py \
  --run-dir "$JOY_RUN" --normalization raw
```

The retained propagation and state-metric dashboards report measured model
state only. Reintroduce behavioral reports with the custom judge's schema.
