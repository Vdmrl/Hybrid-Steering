# Hybrid Steering

Research toolkit for controlling behavior in hybrid language models through
their recurrent state.

Hybrid models such as Qwen3.5 combine attention with recurrent Gated DeltaNet
(GDN) layers. Their KV cache retains addressable context, while the compact
recurrent state can carry persistent behavioral modes. This repository tests
whether those modes can be extracted, edited, combined, and evaluated without
changing the model's factual context.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`steering/`](steering/) | Reusable GDN recurrent-state extraction and intervention primitives |
| [`judge/`](judge/) | Blind LLM-as-a-Judge pipeline for evaluating steering results |
| [`concepts/`](concepts/) | Versioned definitions and anchored rubrics for behavioral features |
| [`experiments/`](experiments/) | Reproducible experiment manifests, small runners, and compact summaries |

The current feature set covers principled candor vs. sycophancy, calm
composure vs. panic, concrete vs. abstract language, and optimism vs.
pessimism.

## Steering

The steering package provides the shared operations needed by experiments:

- discover GDN layers in a hybrid model;
- extract their recurrent state;
- compute a mean direction from paired positive and negative states;
- add a scaled direction or replace selected recurrent states;
- verify that KV and convolution states were not modified;
- save directions with model, layer, shape, and dataset metadata.

Layer indices are absolute, zero-based decoder-layer indices. This avoids the
common ambiguity between “the first GDN layer” and decoder layer `0`.

```python
from hybrid_steering import add_direction, mean_direction, subtract_states

differences = [
    subtract_states(positive_state, negative_state)
    for positive_state, negative_state in paired_states
]
direction = mean_direction(differences)
add_direction(cache, direction, alpha=4.0, layers=[0, 1, 2])
```

See [`steering/README.md`](steering/README.md) for the API and cache safety
checks.

## Judge

Judge v2 evaluates anonymous answers without seeing the steering method,
layer selection, alpha, or condition name.

- Pairwise evaluation is the primary endpoint. Every comparison runs in both
  A/B orders and the two decisions are aggregated into one prompt-level result.
- Scalar evaluation scores one answer per call on an anchored 1–5 trait scale,
  task fulfillment, and coherence.
- Exact answer excerpts are required as evidence.
- Invalid schemas or invented excerpts are retried and persisted as failures.
- Results include prompt/config hashes, answer order, decoding settings, token
  usage, raw responses, and provider response IDs.

Judge v1 remains versioned for reproducing earlier runs. Judge v2 is the
default.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "judge[dev]"
export OPENROUTER_API_KEY="..."

hybrid-judge judge/examples/input.example.jsonl runs/pairwise.jsonl \
  --mode pairwise \
  --feature optimism
```

See [`judge/README.md`](judge/README.md) for input/output schemas, scalar mode,
calibration, and resume behavior.

## Development

```bash
pip install -e "judge[dev]" -e "steering[dev]"
python -m pytest judge/tests steering/tests
ruff check judge steering
ruff format --check judge steering
```
