# Configured benchmark and Judge runs

This experiment reuses `hybrid-direction`, Qwen's `Runner`, the shared recurrent
state math, and the repository's Judge. One JSON plan names a `judge_dataset`
and a `bench_dataset` (IFEval or HumanEval), plus the model and steering
conditions. Qwen uses GDN and Falcon-H1 uses Mamba; the cache adapter differs.

Install with `uv sync --extra dev --extra benchmarks`. Copy one of the
`*.example.json` files, set dataset and direction paths, and replace the
SHA-256 placeholders. The Judge dataset needs `prompt` and an optional `id` or
`key`; HumanEval also needs `task_id`, `test`, and `entry_point`. Datasets are
not committed here. `judge_dataset` can instead specify `hf_repo`, `hf_file`,
and a pinned commit `revision`, alongside `sha256`; the file is fetched through
the Hugging Face cache.

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
uv run python experiments/pipeline/run.py run --config my-run.json --output runs/my-run --run-judge
```

`run` combines generate, score, and plotting. `generate` and `score` are
available separately. Judge API calls require the explicit `--run-judge` flag;
without it, Judge tasks are prepared offline and saved Judge scores are reused.
Generation resumes by `(condition, scale, key)`. Judge and benchmark answers
have separate content-addressed directories under `runs/my-run/judge/` and
`runs/my-run/benchmark/`. Changing only `bench_dataset` leaves the Judge
directory and its answers intact. Each
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

`score` writes blinded `judge_tasks.jsonl` and separate
`judge_bindings.jsonl`. With `--run-judge`, it reads `judge.config`, runs the
existing rubric, and caches `judge_scores.jsonl` plus `sweep_summary.json`.
`benchmark_scores.json` holds official benchmark scores and confidence
intervals. Once both scores exist, `reports/<judge-id>-<benchmark-id>/` contains
CSV, JSON, and SVG. Judge concept-score means and paired differences from the
baseline use a seeded prompt
bootstrap; binary IFEval prompt and HumanEval rates use 95% Wilson intervals;
IFEval instruction rates use a prompt-cluster bootstrap. These marginal
intervals do not establish significance for differences between conditions.
Set Judge API credentials in the environment; do not put them in JSON.
