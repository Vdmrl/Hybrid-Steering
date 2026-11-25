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
bash directions/fetch.sh
sha256sum -c directions/SHA256SUMS.txt
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
MODEL="$(hf download Qwen/Qwen3.5-9B --revision "$REV")"
python -B -m pytest -q -p no:cacheprovider --basetemp .test-tmp tests/test_contracts.py
```

If the checkpoint is already on the server, set `MODEL` to its local snapshot directory. The runtime loads with `local_files_only=True`. Native CUDA and the 32-layer/24-GDN model layout are checked on startup. Use an environment without the optional FLA or causal-conv1d kernels so that both methods use the same pure-Torch recurrence. The manifest records model metadata hashes, checkpoint shard sizes, software/device identity, and the HF revision when available from a cache snapshot path. The directory path alone is not proof of the revision for a manually copied checkpoint.

`directions/fetch.sh` copies only the five CAA and five HyS files from pinned published Hugging Face dataset revisions into their local runtime paths. On an offline node, transfer the package's two direction subdirectories from a connected machine instead:

```bash
cp -R /path/to/provisioned-package/directions/{caa24,hys} directions/
sha256sum -c directions/SHA256SUMS.txt
```

## Jobs

Jobs 1 and 2 are ready immediately. Run on separate allocated GPUs if running in parallel. If there is only one GPU, run them in sequence using its index for both commands.

```bash
CUDA_VISIBLE_DEVICES=0 bash jobs/run_job.sh caa24 --model "$MODEL" \
  --job jobs/job1_caa24_calib_selection8.json --output outputs/job1 --resume

CUDA_VISIBLE_DEVICES=1 bash jobs/run_job.sh hys --model "$MODEL" \
  --job jobs/job2_hys_ablations_and_ctrl.json --output outputs/job2 --resume
```

| Job | Prompts | Conditions | Generated answers | Token cap |
|---|---|---|---:|---:|
| 1 | selection8 | 5 singleton concepts × 6 natural α values | 240 | 1,024 |
| 1b | independent16 | 5 singletons with selected α | 80 | 1,024 |
| 2 | heldout32 | 3 HyS ablations × 12 conditions + 2 runtime controls | 1,216 | 4,096 |
| 3 | heldout32 | 33 CAA24 RSS conditions + 12 CAA24 raw conditions | 1,440 | 4,096 |
| 4 | simple50 | 3 conditions × each of full HyS and CAA24 RSS | 300 | 4,096 |

The 12-condition slice consists of five singletons, five quadruples, and NFTRP at λ=1 and 0.75. Job 2’s two controls are full HyS baseline and full NFTRP λ=0.75, both on single GPU. Job 3’s RSS slice includes baseline, all 31 nonempty subsets at λ=1, and NFTRP λ=0.75. Job 4 has separate baseline IDs for both methods and the same two NFTRP doses.

Review Job 1 on `selection8`. Record one scalar per concept in `selected_alpha.json`, based on `jobs/selected_alpha.example.json`. Use the exact values from `jobs/calibration_grid.json`. Freeze that file before Job 1b. Use Job 1b as an independent check of trait strength, quality and EOS. Record failures and the selection rule before deciding whether to promote the same gains to the final runs.

```bash
CUDA_VISIBLE_DEVICES=0 bash jobs/run_job.sh caa24 --model "$MODEL" \
  --job jobs/job1b_caa24_val_independent16.json --gains selected_alpha.json \
  --output outputs/job1b --resume

CUDA_VISIBLE_DEVICES=0 bash jobs/run_job.sh caa24 --model "$MODEL" \
  --job jobs/job3_caa24_final_template.json --gains selected_alpha.json \
  --output outputs/job3 --resume

CUDA_VISIBLE_DEVICES=0 bash jobs/run_job.sh job --model "$MODEL" \
  --job jobs/job4_external_lambda_simple50.json --gains selected_alpha.json \
  --output outputs/job4 --resume
```

`--check-config` verifies a job and the relevant direction files on CPU. `--max-new-tokens` is available for an explicitly separate run. The default for CAA24 is natural norms and continuous single-token intervention. `--norm-mode unit` uses unit directions and gain divided by √24. `--use-windows` enables N through call 28 and P through call 32, starting with bridge call 1. `--composition-mode` and `--hys-mode` override single-mode configs. Mixed-mode jobs already specify their mode for each condition.

## Calibration and intervention contract

The six-point natural grid is frozen per feature in `jobs/calibration_grid.json`. It includes lower doses for screening without windows. Historical reference-layer norms and gains are retained as context. The JSON also supplies measured `alpha * ||d_l||` at each grid endpoint for L0, L8, L16 and L30. Select doses on dev only.

CAA leaves bulk prefill untouched. The additive intervention applies to the 24 selected decoder blocks on the newline bridge and cached decode forwards. With windows enabled, RSS is recomputed for each active-set phase. Natural mode scales all layers of a concept by one scalar and preserves relative norms.

HyS uses the archived split prefill. Its first recurrent write occurs **after** the bulk forward and **before** the newline bridge. Full, no-RSS and no-Gram then apply the per-concept β after the bridge and each decode forward. Single-write keeps that initial write, then leaves cache evolution to the model. The no-Gram variant shares the first write and uses independent targets from its projected result. This isolates the correction solver, including cross terms created by the common initial write. The other variants keep the original regularized Gram solver.

## Outputs and resume

Return each job’s `generations.jsonl`, `manifest.json`, `judge_input.jsonl`, and `judge_input_manifest.json`, plus `selected_alpha.json` and the dev selection notes. Return partial outputs with their manifest if the job is interrupted. Keep EOS, length stops and empty model outputs. `generations.jsonl` contains condition and question IDs, token IDs and write/finite audits. Both frozen Judges accept the separate shuffled `judge_input.jsonl` directly.

Re-run the same command with `--resume`. It skips completed `(condition_id, question_id)` pairs and checks frozen input, code, direction and runtime fingerprints. Use one process per output directory. An OS-killed process may leave `run.lock`. Confirm its PID is gone before removing the stale lock. If a crash breaks the last JSONL line, preserve a backup and repair the incomplete final line before resume.

## Source provenance

The package contains local ports of the relevant functions. These source paths are relative to the original project root and are provided for review.

- HyS clamp, regularized inverse and schedule: `.worktrees/caa-fr-p18w32-final-20261011/experiments/gdn-rank1-clamp-rss/runtime.py` and `gdn_hook.py`. Cache traversal and recurrent-state access derive from that worktree’s `src/hybrid_steering/cache.py`.
- HyS raw ablation: `.worktrees/qwen9b-gdn-rss-ablation-dev8-20261007/experiments/qwen9b-gdn-rss-ablation-dev8-20261007/search.py`.
- CAA normalization, hooks and native runtime: `.worktrees/caa-fr-p18w32-final-20261011/experiments/caa-fr-p18w32-final-20261011/{search.py,temporal_hook.py,caa_hook.py,native_run.py,run.py,colleague.py}`. The decoding loop is explicit in the port.
- Direction origins and checksums are in `directions/manifest.json`. Prompt banks come from `local/tuning-history-9b-20261010/neutral-prompts.jsonl` and the reference `experiments/forgetting/questions.py` in the CAA worktree. `simple50` uses sampling seed 42. IDs and normalized prompt stems are disjoint across all four packaged banks.

Local verification and GPU limits are recorded in `VERIFICATION.md`.
