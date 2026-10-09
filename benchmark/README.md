# Language-rate Pareto benchmark handoff

This self-contained benchmark directory belongs to `Vdmrl/Hybrid-Steering`.
Run its launcher from the repository root with `bash benchmark/run.sh`, or from
this directory with `bash run.sh`. Its separate virtual environment pins the
local benchmark implementation; the repository's existing `hybrid_steering`
package and historical `experiments/pipeline` results stay separate.

One command runs a frozen Russian/French/Arabic sweep on Qwen3.5-9B and
Falcon-H1-7B-Instruct, selects five strengths per method, then evaluates the
selected strengths on IFEval and HumanEval. Results and model caches stay under
`data/` by default. Generation and scoring resume from exact saved keys.

## One command

On a Linux CUDA machine with Python 3.12, two GPUs that can hold the selected
model, `nvidia-smi`, `bubblewrap` (`bwrap`), and internet access to install
dependencies, tokenizer data, and the model repositories:

```bash
bash run.sh --concept-root /path/to/hybrid-steering-concepts
```

The colleague supplies the separately cloned Hugging Face dataset repository.
The launcher reads only these three files from it:

```text
concepts/en-ru/data/pairs.jsonl
concepts/en-fr/data/pairs.jsonl
concepts/en-ar/data/pairs.jsonl
```

The command installs this bundle into `.venv`, stages the included frozen
benchmarks and evaluator, derives 500 training prompts per language from the
English side of the concept pairs, extracts directions for both models, then
runs the sweep and full benchmarks. Model weights, concept artifacts, results,
and API keys are not in this archive. **No Judge API is used**: the X-axis is
the deterministic `langdetect` language rate.

For another data disk use `--data-root /absolute/path`; for Python from a custom
environment set `PYTHON_BIN=/path/to/python3.12`. Before starting a long run,
check the GPU IDs, model revisions, and memory placement in
`config/language_pareto.json`, `config/qwen-placement.json`, and
`config/falcon-h1-placement.json`. The default IDs are physical GPUs 0 and 1.
An offline preflight, without loading either model, is:

```bash
PYTHONPATH=src:. python3.12 -m experiments.language_pareto_handoff plan \
  --concept-root /path/to/hybrid-steering-concepts
```

The experiment is large: per model and language, the screen contains 96
steered conditions plus baseline, each with 100 responses. The selected stage
has 40 steered conditions plus baseline, each with 500 independent language
prompts, 541 IFEval prompts, and 164 HumanEval tasks. There are six model and
language combinations. Plan for substantial GPU time and disk space.

## Frozen protocol

Methods per model: recurrent full rank, rank 1, rank 2, and rank 1 hard clamp;
residual add and residual projection clamp on each of L16 and L20. On Falcon,
the recurrent state is Mamba, not GDN. Additive recurrent methods intervene
once immediately before the last prompt token. Recurrent clamp fixes its
rank-1 projection before the first output and during every output token.
Residual methods act at the same output-token positions, skipping split
prefill. Thinking is disabled, there is no system prompt, and decoding is
greedy. Truncated recurrent directions are matched to each full direction's
per-head Frobenius norm; this is recorded in the condition identity.

The strength grids are exactly:

```text
recurrent: 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4
residual:  0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.55 0.7
```

The 100-prompt screen has a 512-token output limit. The five distinct measured
strengths closest in total absolute language-rate error to 0.3, 0.5, 0.7,
0.9, and 0.99 are selected separately for each method and layer, in increasing
strength order. Unreachable targets remain visibly unmatched; no interpolated
strength is run. Selection never reads IFEval or HumanEval scores.

The selected strengths run on an independent 500-prompt language set at 512
output tokens and the full IFEval and HumanEval datasets at 2048 output tokens.
The final Pareto X-axis uses the 500-prompt language rate, not the noisier
100-prompt screen. `langdetect` uses seed 0; unknown and empty outputs stay in
the denominator. IFEval Y is official prompt strict; HumanEval Y is pass@1
from code executed inside `bwrap`. Marginal 95% Wilson intervals are shown.
The Pareto frontier contains only measured, nondominated mean points; joining
segments are visual guides, not measurements or a simultaneous confidence band.
Language forcing may break English-language IFEval instructions, so interpret
the Y-axis as benchmark performance under this exact protocol.

Direction extraction uses the same 500 selected English OPUS prompts for both
models. Each model generates a neutral and a target-language answer, with
target-minus-neutral direction. This is a **new frozen extraction protocol**;
the Qwen rank-1 `.safetensors` files in the concept repository cannot supply
the full-rank and rank-2 directions required here. The language-specific
instruction, pair hashes, model revision, code hashes, and task hashes are
saved with the run. The source pairs are not copied into this archive.

Residual add and residual clamp use the same raw mean target-minus-neutral
vector `v` at L16 and L20. Let `u = v / ||v||` and let `h` be the current
residual activation. The methods follow the formulas in the existing
`Hybrid-Steering/experiments/pipeline/residual.py`:

```text
residual add:   h' = h + alpha * v
residual clamp: h' = h - dot(h, u) * u + alpha * v
```

The saved target and neutral mean projections are used only to recover
`||v|| = target_projection - neutral_projection`; the neutral projection is
not added to the clamp destination. Clamp sets `dot(h', u) = alpha * ||v||`
and leaves orthogonal components intact. Both residual methods use the
bundle's output-token hook schedule; split prefill is skipped. The
zero-intervention baseline is a separate condition. Recurrent hard clamp
uses the local projection operator; there is **no separate beta** in this
protocol. Do not label these runs as the historical alpha/beta RSS schedule.

## Files and outputs

- `src/hybrid_steering/`: model loading, extraction, steering, generation,
  official scoring, resume, language detection, plots.
- `experiments/language_pareto_handoff.py`: one-command orchestration.
- `experiments/optimize_benchmarks.py` and `experiments/queue.py`: selected
  strength workflow and GPU queue.
- `config/`: grids, model revisions, GPU placement.
- `data/cache/`: frozen 100/500 prompt banks, IFEval, HumanEval, official
  IFEval evaluator.

Per-model results are written under
`data/results/<model>/<language>/optimization/language-pareto-v1/`. Inspect
`selection.json`, `screen/`, `full/`, `ifeval.csv`, `humaneval.csv`,
`ifeval.png`, and `humaneval.png`. Exact source hashes and progress markers
are stored beside the answers. Resume with the same command and the same
config. Change the study ID for a new protocol; never overwrite saved answers
with different methods, model revisions, prompts, or token limits.

## Prompt for Claude

Copy this prompt to Claude on the colleague's machine:

> You are in the `Hybrid-Steering/benchmark` directory. Read `README.md`,
> `config/language_pareto.json`, and `experiments/language_pareto_handoff.py`.
> The concept repository is at `/REPLACE/WITH/ABSOLUTE/PATH` and has
> `concepts/en-ru`, `concepts/en-fr`, and `concepts/en-ar`. Check Python 3.12,
> CUDA GPUs and free VRAM, the GPU IDs and placement files, `bwrap`, and model
> access. Run the offline `plan` command first and report the six planned
> studies and exact input hashes. Then run
> `bash run.sh --concept-root /REPLACE/WITH/ABSOLUTE/PATH` and monitor progress.
> Resume with the same command if interrupted. Do not change the frozen grids,
> ranks, layers, normalization, prompt sets, model revisions, decoding, token
> limits, or evaluators. Do not launch extra experiments. When complete,
> confirm every condition has all expected prompt keys, then summarize the
> selected strengths and link the six pairs of Pareto plots and CSVs. If a
> preflight check fails, stop and report the exact failure before generating.
