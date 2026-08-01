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
| [`experiments/`](experiments/) | Reproducible manifests, small runners, and compact summaries |

## Steering

The steering package discovers GDN layers, extracts recurrent state, computes
mean directions from paired states, and applies additive or replacement
interventions without changing KV or convolution state. Decoder-layer indices
are absolute and zero-based.

```python
from hybrid_steering import add_direction, mean_direction, subtract_states

differences = [
    subtract_states(positive_state, negative_state)
    for positive_state, negative_state in paired_states
]
direction = mean_direction(differences)
add_direction(cache, direction, alpha=4.0, layers=[0, 1, 2])
```

See [`steering/README.md`](steering/README.md) for the API and cache checks.

## Judge v3

Judge v3 sees only a scenario and an anonymous answer. It does not see the
steering method, layer, rank, alpha, or condition name.

It scores each answer independently on an anchored 1–5 feature scale. The
provider returns one digit; the runner stores the integer score, centered
score, full token-probability distribution, usage, hashes, and provenance.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "judge[dev]"
export OPENROUTER_API_KEY="..."

hybrid-judge judge/examples/input.example.jsonl runs/judgments.jsonl \
  --feature optimism
```

Judge v3 is the only evaluation path and does not use a `--mode` flag. See
[`judge/README.md`](judge/README.md) for schemas, calibration, and resume rules.

## Development

```bash
pip install -r requirements.txt
python -m pytest judge/tests steering/tests
ruff check judge steering
ruff format --check judge steering
```
