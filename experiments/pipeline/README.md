# Configured benchmark and Judge runs

This experiment reuses `hybrid-direction`, Qwen's `Runner`, the shared recurrent
state math, and the repository's Judge. The same JSON plan runs Qwen3.5 GDN or
Falcon-H1 Mamba steering. Only the recurrent cache adapter differs. It supports
IFEval, HumanEval, and a JSONL pool of prompts for Judge-only concept tests.

Install with `uv sync --extra dev --extra benchmarks`. Copy one of the
`*.example.json` files, set absolute dataset and direction paths, and replace
the SHA-256 placeholders. A prompt pool needs `prompt` and an optional `id` or
`key`; HumanEval rows also need `task_id`, `test`, and `entry_point`.

Create a direction from the concept dataset (or pass `--jsonl` for local pairs):

```bash
uv run hybrid-direction --model Qwen/Qwen3.5-9B --concept fairytale \
  --source neutral --target fairytale --pairs 500 --output runs/directions/fairytale
uv run hybrid-direction --model tiiuae/Falcon-H1-7B-Instruct --concept fairytale \
  --source neutral --target fairytale --pairs 500 --output runs/directions/falcon-fairytale
```

`positive_text` is the target, `negative_text` the source. Both artifacts hold
full target-minus-source matrices and mean states; `rank` in the plan is applied
when steering. Keep extraction and evaluation prompts disjoint.

```bash
uv run python experiments/pipeline/run.py generate --config my-run.json --output runs/my-run
uv run python experiments/pipeline/run.py score --config my-run.json --output runs/my-run
uv run python experiments/pipeline/run.py score --config my-run.json --output runs/my-run --run-judge
```

`run` combines generate and score. Generation resumes by `(condition, scale,
key)` and refuses changed config, dataset, direction, or core code. Each
condition has a unique `name`, a `method` (`baseline`, `gdn_add`, `gdn_clamp`,
`mamba_add`, or `mamba_clamp`), a `scales` list, and, when steered, a `direction`
path. `rank`, `normalize`, and `prompt_position` are explicit. Mamba currently
accepts prompt position `-1` (before final prompt token) or `null` (a single
cache update after the prompt; the first generated token is unaffected).
`normalize: false` uses the raw direction. Qwen's existing clamp uses
its rank-1 factor clamp; Falcon's clamp uses the shared target projection in
`state.clamp_delta`. Keep them separate when comparing methods.

IFEval requires Google's `instruction_following_eval` checkout in
`benchmark.evaluator.root` and its `evaluation_lib.py` SHA-256. HumanEval uses
the repository's small executor inside Bubblewrap (`bwrap`) with network and
host writes disabled. Its input is the standard HumanEval JSONL or JSONL.gz.
HumanEval prompts are raw code prefixes; IFEval and Judge prompts use the
model's chat template with thinking disabled.

Without `--run-judge`, `score` writes blinded `judge_tasks.jsonl` and separate
`judge_bindings.jsonl`; it makes no API calls. With the flag, it reads
`judge.config`, runs the existing rubric, and writes `judge_scores.jsonl` plus
`sweep_summary.json`. `benchmark_scores.json` holds official benchmark scores.
Set Judge API credentials in the environment; do not put them in JSON.
