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

### What this study is trying to measure

For each **model × concept × method**, we want a readable Pareto curve:
how much the target concept appears versus how well the model still follows
instructions and solves coding tasks. The currently configured concepts are
Numbered and Theistic, and the models are Qwen3.5-9B and Falcon-H1-7B-Instruct.
The concept pairs are external inputs; this repository does not include their
model weights or the IFEval/HumanEval datasets. The direction sign is always
`target − source`. Numbered uses neutral → numbered pairs; Theistic uses
atheistic framing → theistic framing pairs. The two concepts have independent
directions, Judge rubrics, selections, and reports.

The baseline has scale zero and performs no intervention. Qwen methods are
GDN rank-1 add, rank-2 add, rank-1 clamp, and full-rank add. Falcon uses the
corresponding Mamba recurrent-state methods; a Falcon result must not be called
GDN. Both models also use residual add and residual clamp at decoder layers
**L16 and L20**. Residual clamp uses the actual direction vector `v` and
`u = v / ||v||` to set the component along that axis on every generated token:

```text
residual add:    h' = h + alpha * v
residual clamp:  h' = h - (h · u) * u + alpha * v
```

There is no estimated neutral mean projection in this residual-clamp formula.
The recurrent-state add and clamp operations continue to use the repository's
existing `Runner`/`MambaRunner` and state algebra. For a truncated recurrent
direction, `gain = ||Delta_full||_F / ||Delta_rank||_F` is recorded in the plan;
the actual multiplier is `scale * gain`. This norm matching is part of the
protocol and must be retained when comparing ranks.

### Two stages, with frozen prompt splits

1. Extract a model-specific recurrent direction and residual direction from
   the configured concept pairs. The screen and confirmation Judge prompts
   are disjoint from each other and from extraction prompts.
2. Screen every condition on **50 answers**, greedy decoding, thinking
   disabled, at most **512 new tokens**. All recurrent methods use
   `0.75, 1.00, ..., 4.00` (step 0.25; 14 scales). Each residual method/layer
   uses `0.1, 0.2, ..., 0.9` (9 scales). This is 93 conditions and 4,650
   generated answers per model × concept, including one baseline.
3. The blind Judge returns an integer 0–4. Map a raw score to a concept value
   `c(s) = 0` for 0, `0.5` for 1–2, and `1` for 3–4. The screen concept rate
   at strength `a` is `C(a) = (1/50) * sum_i c(s_i(a))`. Raw scores are kept.
4. For **each method and layer separately**, choose four distinct positive
   scales whose rates are closest in absolute value to `(0.3, 0.5, 0.7, 0.99)`.
   The selector considers scales up through the **first** maximum concept
   rate, requires four distinct nonzero rate levels, minimizes
   `sum_j |C(a_j) - target_j|`, then prefers a larger minimum gap between
   successive selected rates. If no four such levels exist, the study stops
   with an explicit error. It does not invent an extrapolated strength.
5. Run the selected four scales plus the baseline on a separate **50-prompt**
   Judge set with at most **2048 new tokens**. Run the same selected conditions
   on the complete IFEval and HumanEval inputs at the same token limit. This
   is 33 conditions per model × concept. A selection is frozen in
   `selection.json` with the hash of its screen scores.
6. Score saved answers and build the existing Pareto report. IFEval uses the
   pinned official evaluator; HumanEval uses the repository's sandboxed
   execution. Judge scoring is independent of benchmark scoring. The
   original Russian-rate implementation is unchanged.

For **one concept across both models**, the planned screen alone has 9,300
answers. Its confirmation Judge set has up to 3,300 answers and the two full
benchmarks have 33 conditions per model across all their tasks. Plan GPU time
and Judge cost from these counts; a 50-prompt concept estimate is noisy near
0.99, so retain raw labels and uncertainty when interpreting the curve.

### Prepare and run

Copy `pareto.example.json` to a new path and replace every placeholder with
real paths to concept pairs, IFEval input, the Google Research evaluator, and
HumanEval input. You may restrict `models` or `concepts` to one study while
testing. Keep the two bundled 50-prompt files and their contents frozen for a
comparable run. Pin the model snapshots and record dataset hashes before a
production run. Judge credentials belong in environment variables
`OPENAI_BASE_URL` and `OPENAI_API_KEY`, never in the JSON or Git history.

```bash
uv sync --extra benchmarks
uv run python experiments/pipeline/pareto.py \
  --config /absolute/path/to/pareto.json \
  --output /data/your-name/pareto --plan
uv run python experiments/pipeline/pareto.py \
  --config /absolute/path/to/pareto.json \
  --output /data/your-name/pareto --run-judge
```

The second command is the one-command full run after input setup. It resumes
matching answer keys and scores. `--plan` only validates input structure and
prints counts; it does not prove that GPU generation works. Run the real smoke
in `tests/smoke_tp.py` before a large queued study. If Judge credentials are
not supplied, omit `--run-judge`: generation prepares Judge tasks and pauses.
Only use `--run-judge` when API spending is authorized.

The multi-GPU path uses two 16 GiB cards. Qwen runs with genuine two-process
PyTorch tensor parallelism for attention/MLP projections; GDN modules and
their recurrent states remain replicated. Both ranks generate the same batch,
compare answer hashes, and only rank 0 writes. Falcon's installed model has
no native TP plan, so its layers are spread across the two cards instead.
These backends are recorded in each plan. The IML Qwen 512-token batch size
**12** comes from a prior two-A4000 measurement. A 2048-token batch has no
equivalent validation, so the IML run starts at **1**; the first real full
batch must be checked for memory and nonempty output. Batch size is a runtime
setting and can be tuned without changing the scientific condition.

`wait_tp.py` can reserve a physical GPU pair by UUID with shared file locks
and a second occupancy check. For the IML Theistic run, it polls physical GPU
1 and 2 every two seconds and runs the configured command only when both are
under 700 MiB used. The command should first run `tests/smoke_tp.py` for each
backend, then `pareto.py`; do not bypass a failed smoke. The queue status is
`queue-status.json` under its `--root`; `waiting`, `running`, `failed`, and
`complete` describe actual execution, not benchmark quality. Resuming uses
the **same** config, output directory, code, and source files. A changed plan
or manifest is rejected rather than silently mixed with old answers.

### Request for a second agent

Pass the following to an agent who has the clone **and the actual prepared
config/output paths**. Ask it to return evidence with file paths and concrete
recommendations. It may repair implementation bugs and add focused tests; it
should propose any change to the scientific protocol separately so old and
new results are not mixed.

> Review the Numbered/Theistic Pareto pipeline in
> `experiments/pipeline/pareto.py`, `run.py`, `residual.py`, `report.py`,
> `src/hybrid_steering/{runtime,runner,mamba,scoring}.py`, and this README.
> Trace one screen condition from pairs through direction extraction, model
> loading, steering, saved answers, blind Judge, four-point selection, full
> IFEval/HumanEval, and report. Verify that no baseline or prompt is scored
> under the wrong condition, the 50+50 Judge splits and extraction pairs do
> not overlap, `target − source` has the intended sign, residual clamp matches
> the equation above, GDN rank normalization and clamp timing are preserved,
> and the selected four strengths really minimize absolute error to
> 0.3/0.5/0.7/0.99 under the stated constraints. Check that code blocks and
> unfinished/truncated answers do not create misleading Theistic/Numbered
> ratings. Audit TP rank agreement, Falcon layer placement, model revisions,
> OOM behavior at 512 and 2048 tokens, result manifests, resume behavior,
> Judge API usage/cost, official benchmark scoring, and Pareto plotting.
> Inspect a representative real batch and report output quality, token count,
> peak VRAM, and useful progress before calling a long run healthy. Quantify
> uncertainty from only 50 Judge prompts and flag any method whose selected
> rates collapse or miss the targets. Return: (1) confirmed facts with paths,
> (2) concrete bugs with minimal fixes/tests, (3) scientific concerns that
> need a protocol decision, and (4) exact resume commands. Do not silently
> change the metric, rubric, prompt splits, strengths, thinking mode, or
> already saved answers.
