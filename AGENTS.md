# Shared instructions for coding agents

These instructions are tracked because every agent working in this repository
must follow the same contracts. Personal preferences belong outside the
repository or in ignored `AGENTS.local.md`.

## Current scope

The shared component is `judge/`. Keep it independent from a particular
steering implementation, model architecture, or ablation. Experiment-specific
code may consume Judge inputs and outputs, but Judge must not import experiment
code.

Before changing Judge, read:

1. `judge/README.md`;
2. `judge/config/features.yaml`;
3. the affected schema in `judge/schemas/`;
4. `CONTRIBUTING.md`.

## Sources of truth

- Concept meanings live in `judge/config/features.yaml`.
- Runtime defaults live in `judge/config/judge.yaml`.
- Prompt text is versioned in `judge/prompts/`.
- Machine-readable interfaces live in `judge/src/hybrid_judge/models.py`.
- Tiny synthetic examples live in `judge/examples/`.

Do not duplicate a feature definition inside Python code. Do not silently edit
an existing prompt version after it has produced reported results; add a new
version instead.

## Reproducibility rules

- Evaluation must be blind: the judge must not see intervention or method
  names.
- Shuffle answer order deterministically and save the seed or permutation.
- Save judge model, prompt version, rubric version, decoding parameters, and
  token usage with every run.
- Resume by stable `prompt_id` and `answer_id`; never rely on row order.
- Never feed one evaluated answer into another model generation.
- Tests must use fixtures or mocked provider responses. Unit tests must not
  spend API credits.
- Keep raw large generations and run artifacts out of Git. Commit schemas,
  configs, small fixtures, and compact summaries.

## Secrets and external services

- Never commit `.env`, API keys, passwords, proxies, or server addresses with
  credentials.
- Read `OPENROUTER_API_KEY` from the environment.
- Do not print secrets in logs or exception messages.
- External calls require an explicit CLI action; imports and tests must be
  side-effect free.

## Change discipline

- In a shared repository, create a dedicated branch before modifying code,
  prompts, rubrics, schemas, or experiment configs.
- Never commit experimental work directly to `main`.
- Do not continue, rewrite, rebase, or force-push another agent's branch
  without explicit coordination.
- Use `exp/<concept>-<ablation>` for an experiment/config/result bundle and
  the standard `feat/`, `fix/`, `docs/`, `refactor/`, `test/`, or `chore/`
  prefixes for repository code changes.
- Keep one experiment or one logical code change per branch. If an experiment
  requires a reusable Judge feature, prefer a separate `feat/...` branch and
  merge it before the `exp/...` branch consumes it.
- Commit only configs, manifests, small fixtures, and compact summaries from
  experiments. Large generations and model artifacts stay outside Git.
- Make one logical change per commit.
- Use Conventional Commits as documented in `CONTRIBUTING.md`.
- Add or update tests whenever behavior, parsing, schemas, or prompts change.
- A contract-breaking Pydantic model change requires a major rubric version.
- Avoid unrelated formatting or file moves in feature/fix commits.

## Expected checks

Once the Python runner exists, the standard local checks will be:

```bash
python -m pytest judge/tests
ruff check judge
ruff format --check judge
```

If a command is not available yet, do not invent a passing result; report that
the scaffold has no executable implementation.
