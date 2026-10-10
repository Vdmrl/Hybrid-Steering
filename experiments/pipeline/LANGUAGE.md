# Priority 1: Arabic/French language Pareto sweep

This is the experiment to run from the latest commit. Run it before the
older Numbered/Theistic study. No TP, no `torchrun`, no `wait_tp.py`.

## One command

From the repository root:

```bash
bash experiments/pipeline/run_latest.sh
```

The launcher installs the existing environment, prepares NLTK for official
IFEval, downloads only the Arabic/French concept-pair files from
`hybrid-steering/hybrid-steering-concepts`, pins their dataset revision and
both model revisions, and uses the included 50 neutral prompts and official
541-prompt IFEval dataset/evaluator. Existing `.env` or environment Judge
credentials are used; they must never be printed or committed. This command
explicitly enables the existing paid Judge. No runtime Adapter or manual
path editing is needed.

With an existing concept checkout or a different results disk:

```bash
bash experiments/pipeline/run_latest.sh \
  --concept-root /path/to/hybrid-steering-concepts \
  --output /data/language-priority --gpus 0,1
```

Input validation, without loading the evaluated models or making Judge calls:

```bash
uv run python experiments/pipeline/language.py --plan
```

This preparation can fetch dataset/model metadata; it is not a network-free
command. Re-run the same launch command to resume saved answers. Keep code,
inputs and parameters unchanged within a campaign; use a fresh output folder
after changing the protocol. Run in tmux for unattended execution.

## GPU policy: independent workers, efficient measured batches

The expected machine has H100 GPUs; **inspect `nvidia-smi` rather than assume
the GPU type or memory**. Each worker loads a complete model in bfloat16 on
one GPU. Available GPUs run separate model/language jobs concurrently; a
single GPU runs all four jobs sequentially. The default detector selects idle
GPUs with at least 40GB VRAM and refuses occupied explicitly requested GPUs.
It does not stop other users' processes. No parameters are sharded across GPUs.

Extraction batches use 8 examples. Answer generation starts at batch 16 and
can grow to 128 using measured peak allocated memory and response throughput,
with an 85% projected memory target. CUDA OOM retries the same examples with
a smaller batch and caps subsequent growth. Throughput regression stops
growth. Inspect `batch-profile.jsonl` for actual batch sizes, timings, peak
memory fractions and OOM retries. These are runtime observations, not a claim
that an untested batch size is optimal on every H100/context length.

## Exact scope and selection

- Qwen/Qwen3.5-9B and tiiuae/Falcon-H1-7B-Instruct.
- Arabic and French; all exact grids are in `language-grids.json`.
- Fullrank add, rank1 add, rank1-clamp, residual add at **L16**. For L20,
  pass `--residual-layer 20` with a new output folder.
- 134 positive-scale conditions × the same 50 neutral prompts, plus four
  unsteered baselines. Screen: greedy, thinking off, max 512 new tokens.
- Judge concept expression uses the existing 0–4 rubric: score 0 → 0,
  scores 1–2 → 0.5, scores 3–4 → 1. Raw labels remain saved. Explicitly
  unevaluable outputs count as 0; failed API responses are errors, not zeros.
  This is **Judge expression**, not a substituted Lingua language rate.
- Per model/language/method, select four distinct measured positive scales
  maximizing minimum pairwise expression distance. Prefer the increasing
  envelope; include the lowest scale reaching observed expression=1.
- If 1 is not reached, include the observed maximum. If fewer than four
  envelope points exist, use other measured points. **The series still runs
  IFEval**, with a coverage warning. Prefer distances ≥0.2; keep the best
  measured four even when that distance is unattainable. Ties favor lower scales.
- Freeze `selected-ifeval.json/.csv` before any capability measurement. Run
  only the selected scales and baseline on all 541 IFEval prompts, max 1024
  new tokens. Selection never uses IFEval scores. No HumanEval in this language
  experiment; earlier Numbered/Theistic protocols remain separate.

## Steering formulas and identity

For each layer/head, `D = mean_target - mean_source` and
`D_r = U_r diag(s_r) V_r^T`. Truncate each matrix separately; fullrank keeps D.

```text
add (once before last prompt token): X <- X + c_eff D_r
Qwen clamp: S <- S + U_r (c_eff diag(s_r) V_r^T - U_r^T S)
Falcon clamp: E = D_r / max(||D_r||_F, 1e-8)
              M <- M + c_eff <mean_target - M, E>_F E
residual add: h <- h + c v, at the chosen layer each generated token
```

Clamp uses the existing Qwen Runner and Falcon MambaRunner at each generation
step. Their formulas differ. Baseline bypasses every intervention.
Preserve this branch's rank rescaling: `gain = ||D||_F / max(||D_r||_F, 1e-8)`,
summed over selected layers/heads, and `c_eff = c * gain`. Gain is recorded per
condition and is 1 for fullrank; `normalize=false` means no state-norm matching.
Falcon's per-head unit direction inside its clamp is part of that formula.
Do not transfer coefficient interpretation from experiments with another norm.

## Deliverables

Default output: `runs/language-priority/`. Its `README.md` links the four
completed studies and `all-measurements.csv`. Each model/language folder has:

- `pareto.png` / `pareto.svg`: expression vs IFEval prompt-strict with connected
  measured points and 95% expression CI. Dashed line: unsteered IFEval.
- Six pairwise PNG/SVG comparisons for the four methods.
- `screen.csv`: **every** measured screen scale, including plateaus/degradation.
- `per-answer-expression.csv`: all answers, raw Judge labels and mapped expression.
- `selected-ifeval.json/.csv`: selected scales, coverage flags and unmeasured
  suggestions for extending the grid.
- `ifeval.csv`: all selected benchmark measurements, including confidence bounds.
- `measurements.csv`: all screen points joined with IFEval where measured;
  unselected points retain blank benchmark columns, never interpolated scores.
- Raw answer JSONL, manifests, pinned plans, official benchmark scores and
  batch profiles. `COMPLETE.json` is written only after CSV and plots exist.

Expression on the chart is the 50-prompt screen at max512; capability is
IFEval at max1024. Label them accordingly. Straight segments join measurements;
they do not establish scores between measured points or a confidence band
for the whole Pareto frontier.

## Prompt for Claude

> Выполни эксперимент из последнего коммита этой ветки. Сначала прочитай
> AGENTS.md и experiments/pipeline/LANGUAGE.md. Языковой свип имеет приоритет:
> Qwen3.5-9B и Falcon-H1-7B, Arabic/French, точные сетки language-grids.json.
> Проверь реальные GPU (ожидаются H100), их свободную память и существующие
> процессы. Не используй TP/torchrun: по одной копии модели на GPU, независимые
> workers, эффективный batch с измерением throughput и безопасным OOM retry.
> Используй существующие Judge credentials, не показывай ключи. Запусти сначала
> `uv run python experiments/pipeline/language.py --plan`, затем в tmux
> `bash experiments/pipeline/run_latest.sh`. Этот запуск включает Judge API.
> Не меняй сетки, clamp, rank rescaling, метрику expression, prompt pool или
> decoding. Серии без expression=1 тоже идут в IFEval. Мониторь worker logs и
> batch profiles; при прерывании повтори ту же команду для resume. Старый
> Numbered/Theistic запуск через wait_tp.py сейчас не запускай. В конце проверь
> четыре COMPLETE.json, 50 ответов на каждую screen-точку и 541 на каждую
> выбранную IFEval-точку; передай Pareto PNG/SVG, все CSV и ссылку на общий
> runs/language-priority/README.md. Не объявляй завершение при падении worker.
