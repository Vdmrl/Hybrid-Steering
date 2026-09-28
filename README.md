# Hybrid Steering

One package for steering hybrid language models through their Gated DeltaNet
recurrent state, then scoring the result.

A direction is the mean of `target - source`. Positive `scale` moves the state
toward the target. Natural languages are detected locally. Other concepts use
the blind judge and the definitions in `concepts/features.yaml`.

## Setup

```bash
uv sync --extra dev
```

CUDA kernels (`causal-conv1d`, `flash-linear-attention`) are optional:

```bash
uv sync --extra dev --extra fast-kernels
```

## Use

```python
from hybrid_steering import Runner, collect_direction, gdn_layers, load_runtime

model, tokenizer = load_runtime("tiny")  # or "Qwen/Qwen3.5-9B"
layers = gdn_layers(model)
runner = Runner(model, tokenizer, layers)
collected = collect_direction(
    runner,
    [("Target text.", "Source text.")],
)
runner = Runner(model, tokenizer, layers, collected.delta, normalize=True)
tokens = runner.generate(["Describe the weather."], scale=1.0, prompt_position=-1)
```

`prompt_position` counts real prompt tokens: `0` is the initial state, `-1` is
the last prompt token, `None` leaves the prompt alone. `generation_period`
re-applies the direction during decoding.

## Experiments

Each directory under `experiments/` is one experiment. `--smoke` runs it on the
random tiny model with a handful of examples. See `experiments/README.md`.

## Judge

```bash
export OPENROUTER_API_KEY="..."
uv run hybrid-judge examples/input.example.jsonl runs/judgments.jsonl --feature optimism
```

The judge sees a scenario and an answer. It does not see the steering method.
