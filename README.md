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
| [`concepts/`](concepts/) | Definitions and anchored rubrics for behavioral features |
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

Judge v3 evaluates anonymous answers without seeing the steering method,
layer selection, alpha, or condition name.

- Trait evaluation is the primary endpoint for absolute expression and
  composition. Pairwise A/B evaluation remains a causal robustness check.
- Trait evaluation scores one answer per call on an anchored 1–5 scale and
  stores both the chosen score and its token-probability distribution.
- Exact answer excerpts are required only in the optional audit mode.
- Invalid schemas or invented excerpts are retried and persisted as failures.
- Results include prompt/config hashes, answer order, decoding settings, token
  usage, raw responses, and provider response IDs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "judge[dev]"
export OPENROUTER_API_KEY="..."
# Optional when direct OpenRouter access is unavailable:
# export OPENROUTER_PROXY="http://user:password@host:port"

hybrid-judge judge/examples/input.example.jsonl runs/trait.jsonl \
  --feature optimism
```

This default command independently scores every answer on the anchored 1–5
trait scale. Pairwise A/B evaluation is optional secondary evidence, not the
primary composition metric. See [`judge/README.md`](judge/README.md) for
input/output schemas, calibration, and resume behavior.

## Development

```bash
pip install -r requirements.txt
python -m pytest judge/tests steering/tests
ruff check judge steering
ruff format --check judge steering
```
