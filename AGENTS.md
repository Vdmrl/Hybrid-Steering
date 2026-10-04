# Shared instructions for coding agents

One Python package, `hybrid_steering`, installed with uv from the repository
root. Experiment scripts live in flat directories under `experiments/`. They
may import the package. The package must not import experiment code.

## Layout

- `src/hybrid_steering/state.py` — target-minus-source directions, rank, scale, metrics.
- `src/hybrid_steering/cache.py` — recurrent-state reads and writes. Decoder indices are absolute and zero-based.
- `src/hybrid_steering/runner.py` — prefill and greedy generation.
- `src/hybrid_steering/extract.py` — mean direction from paired texts.
- `src/hybrid_steering/capture.py` — GDN kernel transitions and rank metrics.
- `src/hybrid_steering/scoring.py` — concept hit, quality, repetition, and scale choice shared by sweeps.
- `src/hybrid_steering/detect.py` — Lingua for natural language, binary prompts for other concepts.
- `src/hybrid_steering/judge/` — blind 1–5 judge. It must not see method, layer, scale, or condition names.
- `concepts/features.yaml` — feature definitions.
- `config/judge.yaml` — judge runtime defaults.
- `prompts/` — versioned judge prompts.
- `hybrid-direction` — CLI that writes one target-minus-source direction for a concept.
- `experiments/<name>/run.py` — one experiment command. `experiments/forgetting/` is the filler-decay sweep.
- Steering sweeps add `report.py`, which summarizes rows through `hybrid_steering.report`.
- Artifacts go to `runs/<experiment>/<slug>/` (`rows.jsonl`, `summary.json`, `report.html`). Directions are `runs/directions/<slug>/direction`. Question pools are `runs/pairs/`. `runs/` is gitignored.
- `experiments/archive/` — historical scripts, including toy models, notebooks, and their original plots.

## Direction convention

A direction is the mean of `target - source`. Adding a positive `scale` moves
the recurrent state toward the target concept. Language experiments use the
same convention: Russian minus English steers toward Russian. Do not introduce
a second sign, and do not rename `scale` to alpha, strength, or fraction.

`normalize=True` matches each head's delta to the current state norm. A zero
state keeps the raw delta. `rank` is applied when the direction is loaded for
steering, not baked into the stored matrix, unless an experiment explicitly
asks for a truncated artifact.

## Judge

Non-language concepts are scored by `hybrid_steering.judge.score_steering`.
The rubric is `prompts/steering_judge.txt`. Each concept guide is the `guide`
field in `concepts/features.yaml`. One call returns concept presence 0–4 and
content quality 0–4. The judge does not see method, layer, or scale. Requests
go out together through LiteLLM. The endpoint is `OPENAI_BASE_URL` and the
key is `OPENAI_API_KEY`, for OpenRouter or a self-hosted vLLM server. Optional
`OPENROUTER_PROXY` is read from the environment. Do not print them.

Features in `LANGUAGE_FEATURES` use Lingua and do not call a model. Do not copy
a concept guide into experiment code. Extra scores, such as answer equivalence,
stay next to the experiment that needs them.

## Dependencies

```bash
uv sync --extra dev
uv sync --extra dev --extra fast-kernels   # causal-conv1d and flash-linear-attention
```

`fast-kernels` is optional because those wheels need CUDA. They are declared in
`pyproject.toml`. Tests and the tiny-model smoke path do not need them.

## Checks

```bash
uv run ruff check src tests experiments
uv run ruff format --check src tests experiments
uv run pytest
```

## Change discipline

- Work on `main`. Do not create a branch unless the user asks.
- Do not commit `.env`, API keys, raw generations, or model weights.
- Read `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and optional `OPENROUTER_PROXY` from the environment. Do not print them.
- External model calls happen only from an explicit CLI action. Imports and unit tests stay side-effect free.
- Unit tests use the tiny local Qwen or mocked provider responses. They must not spend API credits.
