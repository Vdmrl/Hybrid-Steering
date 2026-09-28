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
- `src/hybrid_steering/detect.py` — Lingua for natural language, binary prompts for other concepts.
- `src/hybrid_steering/judge/` — blind 1–5 judge. It must not see method, layer, scale, or condition names.
- `concepts/features.yaml` — feature definitions.
- `config/judge.yaml` — judge runtime defaults.
- `prompts/` — versioned judge prompts.
- `experiments/<name>/` — one directory per experiment. A historical duplicate keeps a `-legacy` suffix.

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

`concept_detector` is the only concept score. Features in `LANGUAGE_FEATURES`
use Lingua and do not call a model. Every other feature is a 0/1 judge prompt
built from `concepts/features.yaml`. Article-ready 1–5 scores go through
`hybrid_steering.judge`, which calls LiteLLM. Do not copy a feature definition
into experiment code. Answer equivalence is a separate question and is not a
concept detector.

Do not edit a prompt file after it has produced reported results. Add a new
file and point `config/judge.yaml` at it. Read `OPENROUTER_API_KEY` and optional
`OPENROUTER_PROXY` from the environment. Do not print them.

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

Squeezed experiment check, no checkpoint download:

```bash
uv run python experiments/steering/run.py --output artifacts/steering --smoke
uv run python experiments/forgetting/run.py --output artifacts/forgetting --smoke
uv run python experiments/forgetting-squad/run.py --output artifacts/forgetting-squad --smoke
```

## Change discipline

- Before editing a tracked file, run `git branch --show-current`. If it is `main` or `master`, create a branch first.
- One logical code change per branch. An experiment that changes scale, rank, layers, or concepts gets its own `exp/<concept>-<ablation>` branch.
- Use `feat/`, `fix/`, `docs/`, `refactor/`, `test/`, or `chore/` for package changes.
- Do not commit `.env`, API keys, raw generations, or model weights.
- Read `OPENROUTER_API_KEY` and optional `OPENROUTER_PROXY` from the environment. Do not print them.
- External model calls happen only from an explicit CLI action. Imports and unit tests stay side-effect free.
- Unit tests use the tiny local Qwen or mocked provider responses. They must not spend API credits.
