# Qwen3.5-9B GDN: rank1, per-concept clamp and RSS

This is the intervention math and configuration used by the completed
five-concept French composition cohort. It does not launch model generation or
Judge calls. The shared steering package is unchanged.

## Parameters

| Concept | Alpha (before RSS) | Beta (subsequent clamp) |
|---|---:|---:|
| Numbered | 2.5 | 0.25 |
| French | 8 | 0.5 |
| Technical | 2 | 0.5 |
| Theistic | 4 | 0.5 |
| Probabilistic | 1.5 | 0.25 |

All directions are the stored **rank1** recurrent tensors, not CAA vectors.
`run_parameters.json` records absolute zero-based layers, direction paths,
BF16 model / FP32 tensor storage, native TP2 without offload, greedy decoding,
seed, token cap and the split-prefill/newline bridge schedule. `conditions.json`
records all 79 steered conditions; the 32 prompts also have a matching baseline.
Condition alphas already include lambda; do not multiply them by lambda again.

**Alpha** sets the desired displacement in feature coordinates. For a
composition, coefficients are rescaled per layer using RSS:
`sqrt(sum_i ||alpha_i D_i||_F^2) / max(||sum_i alpha_i D_i||_F, 1e-12)`.
This is not normalization to unit norm. Beta is not changed by RSS.

**Beta** controls correction towards the fixed target coordinates after each
subsequent model forward. If `c` is the current coordinate and `t` its target,
the correction is `beta_i * (t_i - c_i)` for each active concept. All corrections
are calculated simultaneously from one state snapshot, using a regularized
Gram inverse (ridge `1e-6`), and mapped back to recurrent state. For correlated
directions and finite precision, the projected change is only approximately
that fraction. KV and convolution states are untouched.

The first intervention uses coefficient **1 for every concept**, then the
selected per-concept betas above. A fresh target/runtime is built for every
answer. This is feedback clamp, not repeatedly adding `alpha * beta * D`.
No unclamped parameter selection was conducted for this cohort; alpha remains
a meaningful parameter in additive steering without clamp.

## Using the runtime

Install the repository dependencies first. When importing from this experiment
directory:

```python
import runtime

# directions[concept][absolute_layer] is the selected tensor, loaded separately.
active = ["numbered", "french"]
alphas = {"numbered": 2.5, "french": 8.0}
betas = {"numbered": 0.25, "french": 0.5}
target = runtime.make_runtime(cache, directions, active, alphas)
runtime.apply_clamp(cache, target, {name: 1.0 for name in active})
# After each subsequent model forward, using that forward's updated cache:
runtime.apply_clamp(cache, target, betas)
runtime.finite_active(cache, target)
```

`gdn_hook.py` is the original TP2 hook reference, called with `base=runtime`.
It checks replicated recurrent state across both distributed ranks and applies
the schedule above. Register a fresh hook for each answer. This bundle is not a
standalone model runner; preserve the recorded generation/cache boundary and
native TP2 layout for reproduction. Moving the clamp to a different prefill
boundary is a different protocol. Merely changing state after a forward does
not recompute that forward's already-produced logits.

## Artifacts and evidence

Tensors, extraction inputs, provenance, stable prompt/answer IDs and all 2560
existing answers are in the pinned dataset release:
https://huggingface.co/datasets/hybrid-steering/hybrid-steering-concepts/tree/0c31436ca565692991e4d5442b3f46843f1102bb/gdn/qwen3.5-9b/rank1-clamp-rss-20261009

Read each concept's `direction_manifest.json` and `hyperparameters.json` before
loading. There are five selected tensors. French uses the existing FLORES128
subset, not OPUS2000. Rank1 is already baked into these artifacts. All 253 token
limit stops and other failures remain in the exported cohort; this release does
not claim globally optimal parameters or universal quality.

The math is copied from the executed composition/runtime functions; the hook
and configuration are copied from the existing export. CPU tests cover distinct
betas, first full correction, zero correction, deterministic replay, RSS routing
and non-recurrent cache preservation. They do not claim a new GPU parity run.

```bash
python -m pytest experiments/gdn-rank1-clamp-rss/test_runtime.py
```
