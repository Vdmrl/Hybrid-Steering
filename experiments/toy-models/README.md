# One-layer Transformer Circuits reproduction

This experiment trains the one-layer model from [A Mathematical Framework for
Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
and extracts its skip-trigram circuits.

## What is reproduced

- One causal, decoder-only attention layer with no MLP.
- `d_model = 12 × 64 = 768`, 2,048-token dense context, and an untied ~50k
  token embedding/unembedding.
- LayerNorm and fixed sinusoidal positions added only to Q and K (the paper's
  Press et al./Shortformer-style position mechanism).
- Bias-free QK, OV, and unembedding paths.

The paper does not publish its exact optimizer, token budget, corpus snapshot,
or checkpoints. This recipe records those choices and substitutes the pinned
[OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) snapshot
for its unpublished Kaplan et al. corpus. The default run trains for 10,000
updates, about 1.31B prediction tokens on one GPU.

## Train

```bash
uv sync --extra toy-models
DATA=data/openwebtext
hf download Skylion007/openwebtext --type dataset \
  --revision 79d93d786212f7344586290adb811d4ae6a1762c \
  --include 'plain_text/train-000[0-1][0-9]-of-00080.parquet' \
  --local-dir "$DATA"
RUN=artifacts/toy_models/one_layer/$(date -u +%Y%m%d-%H%M%S)
CUDA_VISIBLE_DEVICES=0 uv run --extra toy-models python \
  experiments/toy_models/train.py --data-dir "$DATA" --run-dir "$RUN"
```

The recipe targets a high-memory GPU: micro-batch 64, no gradient accumulation,
and bf16. Edit `DEFAULTS` in `train.py` to tune it for different hardware.
`trainer-tools` provides mixed/distributed training, metrics, the cosine
scheduler integration, and resumable checkpoints.

## Gated DeltaNet variants

`model.py` accepts an ordered `ModelSpec.layer_types` tuple. Its GDN blocks
use the GDN-v1 delta-rule recurrence used in Qwen3-Next and remain native
TransformerLens hook modules: use `run_with_cache` and intervention hooks such
as `blocks.0.gdn.hook_q`, `hook_k`, `hook_v`, `hook_beta`, `hook_decay`, and
`hook_state` exactly as for the attention model's cached activations.

```python
ModelSpec(layer_types=("gdn",))          # one GDN layer
ModelSpec(layer_types=("gdn", "attn"))  # GDN then attention
ModelSpec(layer_types=("attn", "gdn"))  # attention then GDN
ModelSpec(layer_types=("gdn", "gdn"))   # two GDN layers
```

Train the one-layer GDN counterpart with the same defaults and checkpoint
format as the attention-only run:

```bash
uv run --extra toy-models python experiments/toy_models/train_gdn.py \
  --data-dir "$DATA" --run-dir artifacts/toy_models/gdn_one_layer
```

The generic trainer also accepts `--layers gdn,attn` (or any of the orders
above). `toy-models` installs Flash Linear Attention, and CUDA GDN runs use its
chunked Triton delta-rule kernel. GPU training errors if that kernel is absent;
pass `--gdn-backend reference` only for a small, inspectable debugging run.
CPU smoke runs automatically use the reference recurrence.

Smoke-test the complete data/training path first:

```bash
uv run --extra toy-models python experiments/toy_models/train.py \
  --data-dir "$DATA" --run-dir /tmp/gdn-one-layer-smoke --smoke
```

Resume an interrupted run without changing its target `--steps`:

```bash
uv run --extra toy-models python experiments/toy_models/train.py \
  --data-dir "$DATA" --run-dir "$RUN" \
  --resume "$RUN/checkpoints/checkpoint_interrupted.pt"
```

Checkpoints, the resolved recipe, and JSONL losses are written under the run
directory. Dataset iteration is deterministic; resuming retokenizes and skips
the already-consumed prefix before continuing.

## Find skip trigrams

```bash
uv run --extra toy-models python experiments/toy_models/skip_trigram_lens.py \
  --run-dir "$RUN"
```

This writes `reports/skip_trigrams.{md,json}`. For every head it selects a
small set of source keys among frequent sampled tokens using the framework
paper's automated heuristic: `max(QK) × max(OV) × token_probability^0.1`.
QK scores are normalized by subtracting the end-of-text key score for each
query; each source's OV logits are mean-centered. It then lists the strongest
queries and outputs for every selected source:

```text
[source] … [destination] → [output]
```

The report also counts matching skip trigrams in a held-out corpus sample. It
is the paper's weight-level content-path lens, not a causal ablation: positional
QK terms and the nonlinear final LayerNorm are reported as limitations.

Extract the final checkpoint's attention patterns for the four fixed causal
probe contexts before rendering the interactive report:

```bash
uv run --extra toy-models python experiments/toy_models/attention_patterns.py \
  --run-dir "$RUN"
```

## Interactive explainer

Render the training curves, checkpoint-level causal probes, and searchable
QK/OV circuit atlas to one self-contained HTML file with
[`research-viz`](https://github.com/ssslakter/research-viz):

```bash
RESEARCH_VIZ_DIR=/path/to/research-viz
"$RESEARCH_VIZ_DIR/.venv/bin/python" experiments/toy_models/interactive_report.py \
  --run-dir "$RUN"
```

The default output is `$RUN/interactive.html`; open it directly in a browser.
It embeds its chart assets and experiment data, so the result needs no server
or network connection.

Run the local check with:

```bash
uv run --extra toy-models python -m unittest discover \
  -s experiments/toy_models -p 'test_*.py' -v
```

[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
provides the attention-only model and exposed QK/OV weights;
[trainer-tools](https://github.com/ssslakter/trainer-tools) keeps the training
loop to experiment-specific code.

## Two-layer induction heads

This reproduces the K-composition induction head from the framework paper and
the phase-change/in-context-learning result from [In-context Learning and
Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html),
using the same shape as the one-layer model (`768 = 12×64`).

`skip_trigram_lens.py` and `attention_patterns.py` are **one-layer only** —
`skip_trigram_lens.py` folds LayerNorm into the embedding, which is only valid
where the residual stream *is* the embedding (layer 0). Use the lenses below
for two layers. All four lenses read attention patterns or QK/OV weights and
raise a clear error on a run with any `gdn` layer; they do not (yet) support
Gated DeltaNet layers.

```bash
RUN=artifacts/toy_models/two_layer/$(date -u +%Y%m%d-%H%M%S)
CUDA_VISIBLE_DEVICES=0 uv run --extra toy-models python \
  experiments/toy_models/train.py --data-dir "$DATA" --run-dir "$RUN" --layers 2
```

Training also retains weights-only snapshots (`snapshots/step_N.pt`) at 32
log-spaced steps, so a checkpoint sweep does not need `--keep-checkpoints` to
hold every full optimizer state.

Sweep every snapshot for behavioral/causal induction diagnostics — attention
to the induction/duplicate/previous-token offsets on a repeated random
sequence, in-context learning score, and per-head zero-ablation:

```bash
for f in "$RUN"/snapshots/step_*.pt; do
  step=$(basename "$f" .pt)
  uv run --extra toy-models python experiments/toy_models/induction_lens.py \
    --run-dir "$RUN" --checkpoint "$f" --output-dir "$RUN/reports/$step"
done
uv run --extra toy-models python experiments/toy_models/induction_lens.py \
  --run-dir "$RUN" --dump-patterns
```

The last invocation (no `--checkpoint`) runs on `model_final.pt` and writes
`reports/final/induction.json`, which `circuit_lens.py` reads to pick the
strongest previous-token and induction heads:

```bash
uv run --extra toy-models python experiments/toy_models/circuit_lens.py --run-dir "$RUN"
```

This writes `reports/circuits.{md,json}`: Q/K/V composition scores between
every layer-0/layer-1 head pair (baseline-subtracted against random
rank-matched matrices), an exact OV copying score per head, and the full
induction QK circuit's diagonal-dominance accuracy.

Render the loss/LR curves, ICL-score-vs-step, per-head induction heatmap,
composition heatmaps, ablation table, and attention patterns to one
self-contained HTML file:

```bash
"$RESEARCH_VIZ_DIR/.venv/bin/python" experiments/toy_models/induction_report.py --run-dir "$RUN"
```
