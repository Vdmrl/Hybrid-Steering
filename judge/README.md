# Hybrid Judge

Blind LLM-as-a-Judge for steering ablations. It receives only a scenario and
anonymous answers: never GDN/residual/SVD, layer, alpha, or method names.

## Judge v2

Three modes are intentionally separate:

- **trait (default):** scores one answer per call on an anchored 1–5 trait
  scale and returns no unrelated quality judgment;
- **pairwise (complementary):** compares A and B in both answer orders. The two calls
  are aggregated into one prompt-level result. An order disagreement becomes a
  conservative tie and is marked inconsistent;
- **scalar (secondary):** scores exactly one answer per call on an anchored
  1–5 trait scale, task fulfillment, and coherence. Trait score 3 means
  neutral, mixed, absent, or unclear; `centered_trait_score = score - 3`.

Every judgment must cite exact answer excerpts. Invalid JSON, unexpected IDs,
or invented excerpts are retried and then written to a failures sidecar.

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

## Run

Complementary pairwise evaluation (both orders are mandatory):

```bash
hybrid-judge judge/examples/input.example.jsonl runs/pairwise.jsonl \
  --mode pairwise --feature optimism
```

This writes:

- `pairwise.jsonl`: raw order-level judgments and full provenance;
- `pairwise.aggregated.jsonl`: one conservative result per prompt/pair;
- `pairwise.failures.jsonl`: append-only failed tasks, if any.

Trait-only scalar evaluation for new experiments:

```bash
hybrid-judge judge/examples/input.example.jsonl runs/scalar.jsonl \
  --mode trait --feature optimism
```

Legacy `--mode scalar` preserves the v2 combined
trait/task-fulfillment/coherence contract. Evaluate answer quality separately
once per answer instead of repeating it for every trait.

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
judge/prompts/scalar_v2.txt
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
