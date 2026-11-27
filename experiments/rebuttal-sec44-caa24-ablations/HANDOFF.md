# §4.4 rebuttal — Qwen3.5‑9B GPU handoff

Run from `experiments/rebuttal-sec44-caa24-ablations/`. The branch includes four fixed prompt banks. Direction tensors and model weights are supplied separately.

## Environment

Use Linux, Python 3.11, CUDA 12.8, a GPU with native BF16, and roughly 24 GiB or more of **free** VRAM. The pin follows the historical single-native CAA environment. An A100/H100/H200 or comparable device with additional headroom is preferable for 4,096-token jobs. Expose exactly one GPU per process. Use pure-Torch GDN recurrence, SDPA, BF16, and the pinned Qwen revision.

```bash
cd /path/to/Hybrid-Steering/experiments/rebuttal-sec44-caa24-ablations
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
bash directions/fetch.sh  # requires hf auth login or an authorized HF_TOKEN
sha256sum -c directions/SHA256SUMS.txt
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
MODEL="$(hf download Qwen/Qwen3.5-9B --revision "$REV")"
python -B -m pytest -q -p no:cacheprovider --basetemp .test-tmp tests/test_contracts.py
```

If the checkpoint is already on the server, set `MODEL` to its local snapshot directory. The runtime loads with `local_files_only=True`. Native CUDA and the 32-layer/24-GDN model layout are checked on startup. Use an environment without the optional FLA or causal-conv1d kernels so that both methods use the same pure-Torch recurrence. The manifest records model metadata hashes, checkpoint shard sizes, software/device identity, and the HF revision when available from a cache snapshot path. The directory path alone is not proof of the revision for a manually copied checkpoint.

`directions/fetch.sh` defaults to the access-controlled Hugging Face dataset `hybrid-steering/hybrid-steering-concepts` (CAA revision `aeff2166d7406fcb59de83e7602f1a736a142a32`, HyS revision `82be2f9a41bdc480a8cd1cb86a06b6fb7978f3e4`). GitHub clone access does not grant HF access. Use a read token from a Hugging Face account with dataset access. On an offline node, point at a staged directory containing `caa24/` and `hys/` (or their parent package):

```bash
bash directions/fetch.sh --from-dir /path/to/provisioned-package/directions
sha256sum -c directions/SHA256SUMS.txt
```

## Jobs

If your team already hosts a GitHub Release mirror, use `bash directions/fetch.sh --github-release OWNER/REPO TAG` with `gh auth login`, `GH_TOKEN` or `GITHUB_TOKEN`. The release must contain `caa24-{N,F,T,R,P}.safetensors` and `hys-{N,F,T,R,P}.safetensors` as ten separate assets. No GitHub mirror is assumed to exist. All three input modes verify every staged file before installing it.

With `GPUS=0`, Drop #1 runs Job 1 mega → Job 2 → Job 4a. With `GPUS=0,1`, Job 1 uses GPU 0 and Job 2 independently uses GPU 1; Job 4a starts after both complete and uses the full list. Pass GPU IDs actually allocated on the host. Each worker loads its own BF16 copy. Treat 4–5 workers on H200 141 GB, 2–3 on A100/H100 80 GB, and one on 40/48 GB as starting points, not measured capacity. Linux workers sharing the same parent output directory serialize model loading; the default launch stagger also avoids starting every Torch import at once. Neither limit guarantees enough RAM. If the scheduler or a worker reports OOM, lower `WORKERS_PER_GPU` and rerun with `--resume`.

```bash
MODEL="$MODEL" GPUS=0 WORKERS_PER_GPU=4 bash jobs/run_drop1_h200.sh
```

**Return `outputs/job1_mega` as soon as Job 1 finishes, while Job 2 is still running.** Job 4a is HyS-only and does not wait for CAA calibration. Restart the same script to resume.

| Job | Prompts | Conditions | Generated answers | Token cap |
|---|---|---|---:|---:|
| 1 mega | selection8 | 13 families, 478 singleton configs + 1 baseline | 3,832 | 1,024 |
| 1 legacy | selection8 | 5 singleton concepts × 6 natural α values | 240 | 1,024 |
| 1b | independent16 | 5 singletons with selected α | 80 | 1,024 |
| 2 | heldout32 | 3 HyS ablations × 12 conditions + 2 runtime controls | 1,216 | 4,096 |
| 3 | heldout32 | 33 CAA24 RSS conditions + 12 CAA24 raw conditions | 1,440 | 4,096 |
| 4 | simple50 | 3 conditions × each of full HyS and CAA24 RSS | 300 | 4,096 |
| 4a | simple50 | full HyS baseline + NFTRP λ=0.5, 0.75, 1 | 200 | 4,096 |

The 12-condition slice consists of five singletons, five quadruples, and NFTRP at λ=1 and 0.75. Job 2’s two controls are full HyS baseline and full NFTRP λ=0.75, both on single GPU. Job 3’s RSS slice includes baseline, all 31 nonempty subsets at λ=1, and NFTRP λ=0.75. Job 4 has separate baseline IDs for both methods and the same two NFTRP doses.

Review Job 1 mega on `selection8`. Copy `jobs/selected_caa_config.json` to `selected_caa_config.json` and fill in the selected family, layers, norm mode, window settings, and all five measured doses. Prefer one family that works across concepts. Windowed families only sweep N and P; use a dose for F/T/R from the matching no-window family (same layers and norm), since the windows never affect F/T/R. Freeze the choice before Job 1b. Use Job 1b to validate trait strength, quality, and EOS independently before promoting the same configuration. The old five-number file still works for the legacy default; it does not select a family.

```bash
python -B jobs/run_parallel.py --runner caa24 --model "$MODEL" --gpus 0 --resume \
  --job jobs/job1b_caa24_val_independent16.json --selected-caa-config selected_caa_config.json \
  --output outputs/job1b

python -B jobs/run_parallel.py --runner caa24 --model "$MODEL" --gpus 0 --resume \
  --job jobs/job3_caa24_final_template.json --selected-caa-config selected_caa_config.json \
  --output outputs/job3

python -B jobs/run_parallel.py --runner caa24 --model "$MODEL" --gpus 0 --resume \
  --job jobs/job4_external_lambda_simple50.json --selected-caa-config selected_caa_config.json \
  --output outputs/job4
```

The launcher detects mixed-method Job 4 and dispatches both methods. Append `--dry-run` to any launcher command to print the shard plan without loading weights or writing outputs. For single-process use, `jobs/run_job.sh` still works and accepts `--check-config` for CPU validation of directions. Job 4a and Job 4 remain distinct runs, each with its own provenance.

## Calibration and intervention contract

The legacy six-point grid and physical doses are still in `jobs/calibration_grid.json`. The 13 core-family grids in Job 1 mega add up to 468 configs. To reach the requested 478 without dropping or changing a core point, the job adds one geometric midpoint between the two lowest doses per trait in each of `gdn24_natural` and `all32_natural` (10 configs, marked `supplemental`). Middle-layer natural grids scale by `24 / |layers|` and clip to the stated trait-specific bounds. The shared baseline is never steered.

CAA leaves bulk prefill untouched. Additions apply only to selected decoder blocks on the newline bridge and decode forwards. Natural mode uses `λ × α × d_l`; unit mode uses `λ × gain / √|layers| × d_l / ‖d_l‖`. RSS acts per layer on those contributions and is exactly 1 for singletons. Windows count from bridge call 1, retaining N through call 28 and P through call 32, with an RSS recalculation after each cutoff. The default remains `gdn24`, natural, and no windows when no family is specified.

HyS uses the archived split prefill. Its first recurrent write occurs **after** the bulk forward and **before** the newline bridge. Full, no-RSS and no-Gram then apply the per-concept β after the bridge and each decode forward. Single-write keeps that initial write, then leaves cache evolution to the model. The no-Gram variant shares the first write and uses independent targets from its projected result. This isolates the correction solver, including cross terms created by the common initial write. The other variants keep the original regularized Gram solver.

## Drop #2

```bash
MODEL=/absolute/path/to/pinned/Qwen3.5-9B/snapshot GPUS=0,1 WORKERS_PER_GPU=2 \
  bash jobs/run_drop2.sh --resume
```

Jobs run in table order, and each job shards its answers across the entire `GPUS` list. HyS workers still run single-GPU; there is no tensor parallelism. For H100 80 GB, start with `WORKERS_PER_GPU=2` for the 4,096-token jobs; reduce to 1 if memory is insufficient. `--resume` is already the default. Append `--dry-run` to print the plan without loading model weights. Return `generations.jsonl`, `blind.jsonl`, and `manifest.json` per completed output; return Job 2b and Job 1b as soon as each finishes.

| Job | Bank | Answers | Cap | Output under `outputs/` |
|---|---|---:|---:|---|
| Job 2b HyS full | heldout32 | 384 | 4,096 | `job2b` |
| Job 1b both CAA selections + baseline | independent16 | 176 | 1,024 | `job1b` |
| Job 4b CAA best, three λ values | simple50 | 150 | 4,096 | `job4b` |
| Job 3 CAA best, factorial + extra λ | heldout32 | 1,088 | 4,096 | `job3_best_multilayer` |
| Job 3b CAA strict, matched 24 GDN layers | heldout32 | 416 | 4,096 | `job3b_gdn24_ctrl` |
| Job 1c 24-layer gap sweep | selection8 | 96 | 1,024 | `job1c_gdn24_micro` |
| **Total** | | **2,310** | | |

Job 2b reuses the Drop #1 Job 2 baseline and λ=0.75 control for reporting; Job 4b reuses the Job 4a baseline. These baselines are not regenerated in the corresponding jobs. Do not pool Job 1c dev tuning with Job 1b independent validation or change these frozen selections based on the same independent evaluation.

The packaged selections in `jobs/configs/` are explicitly referenced by each active CAA condition. A custom selected JSON may instead contain a `per_concept` map with keys `N,F,T,R,P` and entries `layers`, `norm_mode`, `value`, and `window`; pass it to a template via `--selected-caa-config`. The old five-number and homogeneous selection formats remain supported; an output generated with another code hash still requires a separate directory. At layer `l`, RSS combines only concepts assigned to that layer whose windows are still active. A unit dose is divided by the square root of that concept's own layer count. Bridge call 1 is generated-token index 0, so N=28 is off at index 28 (call 29). The raw and merged JSONL use ASCII escapes without changing decoded text or the blind schema.

These are the exact requested frozen settings, not a claim of zero cross-talk. In Drop #1, the selected `all32_unit` F point (gain 1.05893) measured F=91.67%, Q=3.75 and maximum off-target uplift +12.50 pp (T), not 0. It differs from the alternative 24-layer F point in the selection analysis. The other four best-multilayer points have zero maximum measured positive uplift. The strict T control (gain 1 on 24 GDN layers) has no Stage B measurement. No heterogeneous five-way composition was measured in Drop #1. Job 1b provides a new check; it does not automatically retune the other jobs.

## Outputs and resume

Return each completed job’s `generations.jsonl`, `manifest.json`, `blind.jsonl` (identical to `judge_input.jsonl`), and `judge_input_manifest.json`, plus `selected_caa_config.json` and dev selection notes when available. Keep EOS, length stops, and empty model outputs. Resolved CAA settings are in the raw rows and manifest, never in the shuffled blind export. For interrupted jobs, retain the entire directory including `workers/`.

Re-run the launcher with `--resume`. It skips completed `(condition_id, question_id)` pairs from the merged file and all worker files, ignoring only a torn final worker line. It keeps logs and old shards, and validates the frozen input and runtime fingerprint. Worker count and GPU assignment may change across attempts; code and inputs may not. Do not mix output from different commits. Outputs are atomically replaced after coverage checks; `manifest.json` is marked `complete` last. Use one launcher per output directory. A hard kill may leave `run.lock`; confirm its PID **and its workers** are gone before removing the stale root lock. Do not share an output directory between the parallel and single-process runners.

## Source provenance

The package contains local ports of the relevant functions. These source paths are relative to the original project root and are provided for review.

- HyS clamp, regularized inverse and schedule: `.worktrees/caa-fr-p18w32-final-20261011/experiments/gdn-rank1-clamp-rss/runtime.py` and `gdn_hook.py`. Cache traversal and recurrent-state access derive from that worktree’s `src/hybrid_steering/cache.py`.
- HyS raw ablation: `.worktrees/qwen9b-gdn-rss-ablation-dev8-20261007/experiments/qwen9b-gdn-rss-ablation-dev8-20261007/search.py`.
- CAA normalization, hooks and native runtime: `.worktrees/caa-fr-p18w32-final-20261011/experiments/caa-fr-p18w32-final-20261011/{search.py,temporal_hook.py,caa_hook.py,native_run.py,run.py,colleague.py}`. The decoding loop is explicit in the port.
- Direction origins and checksums are in `directions/manifest.json`. Prompt banks come from `local/tuning-history-9b-20261010/neutral-prompts.jsonl` and the reference `experiments/forgetting/questions.py` in the CAA worktree. `simple50` uses sampling seed 42. IDs and normalized prompt stems are disjoint across all four packaged banks.

Local verification and GPU limits are recorded in `VERIFICATION.md`.
