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

## Four-point Pareto sweep

`pareto.py` is a thin coordinator for the pipeline above. Copy
`pareto.example.json`, set the concept pair paths and the existing IFEval and
HumanEval dataset paths, then run:

```bash
uv sync --extra dev --extra benchmarks
uv run python experiments/pipeline/pareto.py \
  --config experiments/pipeline/pareto.example.json \
  --output runs/pareto --run-judge
```

The script calls the existing `hybrid-direction` and residual extraction code,
then `run.py` for generation, Judge scoring, IFEval, HumanEval, and report
artifacts. It runs Qwen3.5-9B and Falcon-H1-7B on Numbered and Theistic. For
each concept and model, rank 1, rank 2, rank 1 clamp, and full rank use scales
0.75–4 in steps of 0.25. Residual add and clamp use 0.1–0.9 in steps of 0.1
at L16 and L20. The screen uses 50 answers per condition with a 512-token
limit. Judge labels 0, 1–2, and 3–4 map to 0, 0.5, and 1. For each method and
layer, the selector chooses four distinct positive pre-peak means that minimize
the total absolute distance to 0.3, 0.5, 0.7, and 0.99. It records each
distance in `selection.json`; if four levels do not exist, the run stops before
the full benchmarks. The selected scales use an independent 50-prompt Judge
set and the full IFEval and HumanEval datasets, with a 2048-token limit. The
two-tier Judge results appear in the existing Pareto report format, while raw
Judge labels remain in `judge_scores.jsonl`.

The included two 50-prompt files are frozen and disjoint. Input file hashes
are recorded in the generated plans. `--run-judge` explicitly permits paid
API calls; without it, generation stops after preparing blinded Judge tasks.
The original Russian, French, and Arabic detection code is unchanged.
