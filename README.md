# Hybrid Steering

One package for steering hybrid language models through their Gated DeltaNet
recurrent state, then scoring the result.

A direction is the mean of `target - source`. Positive `scale` moves the state
toward the target. Natural languages are detected locally. Other concepts use
the steering judge and the guides in `concepts/features.yaml`.

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

## Direction

```bash
uv run hybrid-direction --concept en-ru --output runs/en-ru
```

`--jsonl` reads local pairs instead of the Hub. `positive_text` is the target
and `negative_text` is the source. Experiments load `runs/en-ru/direction`.

## Experiments

Each steering experiment has `run.py` (generate and score) and `report.py`
(summarize the rows). See `experiments/README.md`.

## Judge

Steering runs score non-language concepts with `score_steering`. The judge sees
the prompt and the answer. It does not see the steering method. Set
`OPENAI_BASE_URL` and `OPENAI_API_KEY`. OpenRouter uses
`https://openrouter.ai/api/v1`. A self-hosted vLLM server uses its `/v1` URL.
`config/judge.yaml` is the model id that endpoint expects.
