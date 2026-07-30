# Hybrid Judge

Blind LLM-as-a-Judge for steering ablations. It receives only a scenario and
anonymous answers: never GDN/residual/SVD, layer, alpha, or method names.

## Recommended evaluation

Run Hybrid Judge normally for new experiments; no `--mode` flag is required.
It independently scores every answer on the anchored 1–5 scale. The provider
returns one digit and token log-probabilities; the runner attaches IDs, full
provenance, and a conditional probability distribution over scores 1–5. This
is the inexpensive standard for large ablations and supports both effect
intensity and joint composition.

```bash
hybrid-judge judge/examples/input.example.jsonl runs/trait.jsonl \
  --feature optimism
```

Trait score 3 means neutral, mixed, absent, or unclear;
`centered_trait_score = score - 3`.
`score_distribution.expected_score` is the probability-weighted soft score;
`chosen_score_probability` and `entropy` describe uncertainty. Probabilities
are renormalized over the five valid score tokens, while `valid_token_mass`
records how much of the returned probability mass they covered.

Use `--mode trait-audit` when exact supporting evidence and a short reason are
needed. The audited format costs more and can fail strict quote validation, so
run it on a representative audit sample instead of duplicating a large run.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "judge[dev]"
export OPENROUTER_API_KEY="..."
# Optional:
# export OPENROUTER_PROXY="http://user:password@host:port"
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY="..."
$env:OPENROUTER_PROXY="http://user:password@host:port" # optional
```

## Input

One JSON object per scenario:

```json
{
  "prompt_id": "scenario-001",
  "scenario": "What should the assistant do next?",
  "answers": [
    {"answer_id": "baseline", "text": "First answer"},
    {"answer_id": "steered", "text": "Second answer"}
  ],
  "metadata": {"dataset": "shared-core-v1", "split": "validation"}
}
```

`answer_id` is used only to join results. The provider sees anonymous
`answer_0` or `A`/`B` labels.

## Optional modes

Audited trait evaluation:

```bash
hybrid-judge judge/examples/input.example.jsonl runs/trait-audit.jsonl \
  --mode trait-audit --feature optimism
```

Pairwise A/B evaluation is a complementary causal robustness check. It should
not replace the 1–5 trait scores in new composition experiments. Both answer
orders are mandatory:

```bash
hybrid-judge judge/examples/input.example.jsonl runs/pairwise.jsonl \
  --mode pairwise --feature optimism
```

This writes:

- `pairwise.jsonl`: raw order-level judgments and full provenance;
- `pairwise.aggregated.jsonl`: one conservative result per prompt/pair;
- `pairwise.failures.jsonl`: append-only failed tasks, if any.

For answer-quality checks, use the same default command with
`--feature answer_quality`. This keeps all large evaluations on the same
one-token probabilistic contract.

Useful options:

```text
--workers 8
--seed 20260728
--config-root judge
```

Re-running against the same output resumes by stable `task_id`. The model is
set once in `judge/config/judge.yaml`.

## Versioned sources of truth

```text
concepts/features.yaml
judge/config/judge.yaml
judge/prompts/trait_compact_v1.txt
judge/prompts/scalar_v3.txt
judge/prompts/pairwise_v2.txt
```

Results store model/provider, prompt and config hashes, rubric/config versions,
answer order, seed, decoding settings, usage, schema attempts, raw provider
responses, response IDs, and UTC timestamp.

## Before article use

Read [calibration/README.md](calibration/README.md). A valid API run is only
exploratory until every feature passes the human calibration and order-bias
gates. Trait scores are the primary endpoint for absolute expression and
composition; pairwise comparisons remain a causal robustness check.
