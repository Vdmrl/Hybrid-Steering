# How to contribute

## Branches

Work on `main`. Do not create a branch unless asked.

Do not force-push `main`. A pull request is optional. When one exists, it
should contain the exact config or manifest and a compact summary, not large
generations or weights.

## Commits

We use Conventional Commits:

```text
<type>(<scope>): <short description>
```

Allowed `type` values:

- `feat` — a new capability;
- `fix` — a bug fix;
- `docs` — documentation only;
- `test` — tests and fixtures;
- `refactor` — a structural change with no new behavior;
- `perf` — a speedup or a cost reduction;
- `chore` — dependencies and maintenance;
- `ci` — automated checks.

Main `scope` values:

- `judge` — the shared pipeline;
- `steering` — reading and writing recurrent state;
- `artifact` — direction formats and run manifests;
- `experiment` — the reproducible config of one experiment;
- `rubric` — scales and definitions;
- `prompt` — judge prompts;
- `runner` — the queue, retries, and resume;
- `docs`, `ci`.

Examples:

```text
feat(judge): add provider metadata
feat(rubric): add optimism versus pessimism scale
fix(runner): resume partially judged prompts
docs(prompt): explain anchored score meanings
```

Write the subject in the imperative mood, with no trailing period. Do not use
`update`, `changes`, `work`, or `wip` as the description of a finished commit.

Put the reason and important decisions in the commit body:

```text
feat(prompt): add answer-order reversal check

Run every comparison in both orientations to detect position bias.
Store both raw decisions before aggregation.
```

## What a pull request should contain

- a short goal;
- the commit or range that contains the change;
- which files are the sources of truth;
- how the change was checked;
- whether the judge cost changes;
- a small input and output example for new behavior.

Do not add to a pull request:

- `.env` files and tokens;
- full model generations;
- large datasets and weights;
- incidental local reports;
- an unrelated refactor together with a new metric.

## Changing the steering judge

The rubric is `prompts/steering_judge.txt`. Concept guides are the `guide`
fields in `concepts/features.yaml`. Language detection does not use that rubric.
An extra score for one experiment lives next to that experiment.
