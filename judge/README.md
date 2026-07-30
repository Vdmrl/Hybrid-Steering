# Hybrid Judge v3

Blind LLM-as-a-Judge for steering ablations. It sees only a scenario and an
anonymous answer, never the steering method, condition, layer, rank, or alpha.

Judge v3 is the only evaluation path. It independently scores every answer on
the feature's anchored 1–5 scale. The model returns one digit; the runner adds
stable IDs, provenance, token use, and a probability distribution over scores
1–5 from token log-probabilities.

```bash
hybrid-judge judge/examples/input.example.jsonl runs/judgments.jsonl \
  --feature optimism
```

There is no `--mode` flag.

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
  "metadata": {"dataset": "shared-core", "split": "validation"}
}
```

`answer_id` joins results but is never shown to the provider. Every answer is
evaluated independently as `answer_0`.

## Output

Each result contains:

- `trait_score`: integer 1–5;
- `centered_trait_score`: score minus 3;
- `score_distribution.expected_score`: probability-weighted soft score;
- probabilities for every score, chosen-score probability, entropy, and valid
  score-token mass;
- full model, prompt, config, usage, response, and timestamp provenance.

Score 3 means neutral, mixed, absent, balanced, or unclear. Use integer scores
as the primary endpoint. Treat the more sensitive expected score as secondary
until it has its own human calibration.

For compositions, judge every active feature separately and report both the
per-feature paired changes and joint metrics such as the proportion of answers
where every active feature scores at least 4. Evaluate answer quality once per
answer with `--feature answer_quality`.

Useful options:

```text
--workers 8
--seed 20260728
--config-root judge
```

Re-running against the same output resumes by stable `task_id`. Resume is
rejected when model, prompt, rubric, config, seed, or decoding provenance does
not match. The model is set once in `judge/config/judge.yaml`.

## Sources of truth

```text
concepts/features.yaml
judge/config/judge.yaml
judge/prompts/judge_v3.txt
```

Read [calibration/README.md](calibration/README.md) before using results in an
article. A valid API response is not proof that the Judge agrees with humans.
