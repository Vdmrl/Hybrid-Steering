# Frozen French CAA composition experiment

Run ordinary additive CAA for Qwen/Qwen3.5-9B, pinned revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, native BF16 on ONE CUDA GPU.
No training, no Judge/API calls. Rank and clamp beta do not apply to CAA.
All five supplied mean-content directions use all256 training pairs and are
unit-normalized per layer. Gains below are multiplied by lambda and divided
by sqrt(number of selected layers); there is NO cross-concept RSS normalization.

| Concept | zero-based layer | gain at lambda=1 | active single-token forwards |
|---|---|---:|---|
| Numbered | 8,16 |16|first28|
| French |8|4|continuous|
| Technical |16|8|continuous|
| Theistic framing |8|12|continuous|
| Probabilistic framing |8|14|first24|

Bulk prefill is untouched. Forward count includes the final newline bridge.
System prompt, greedy decoding, seed20260911, cap4096 and all32 prompt IDs
are in the task files. Do not alter them or tune on these heldout answers.
Prob remains imperfect: prior dev primary4/8; both new delayed-onset checks
primary4/8, advice0/4 and punctuation failures8/8. These are manual presence
counts, not Judge scores or uplift. Frozen selection includes reviewed IDs/hashes.

## Work split

Owner RTX5090: `tasks/core32.json`, all31 nonempty subsets at lambda1:
992 steered answers plus32 baseline =1024 main observations.
COLLEAGUE H200: **ONLY `tasks/appendix32.json`**,48 fixed lambda conditions,
1536 steered answers. It also generates32 fresh baselines,10 zero controls
and48 replay controls. These81 controls are not extra main observations.
Together the main design is32x80=2560. Do not duplicate the core job.
The task already contains the exact lambda values/gains; do not multiply again.

## Setup and exact command

Use an isolated environment, Python3.12, torch2.7.1+cu128,
transformers5.14.1, safetensors0.8.0, accelerate1.15.0. Other environments
are separate runtime cohorts and must be recorded. The RTX5090 probe used
pure torch recurrent kernels without fla/causal-conv1d. Do not silently mix
kernel implementations or claim identical answers across GPUs.

Download the pinned model through Hugging Face `snapshot_download` or use an
already cached matching revision. No private credentials are required for this
public model. Supply the model directory in `STEERING_MODEL_PATH`.

The five vectors are supplied separately in `caa-directions-20261011.tar.gz`
(large model artifacts stay outside Git). Unpack the archive in THIS experiment
directory: it creates `directions/*.safetensors`. The runner verifies each
actual SHA256 against frozen-config.json before generation. Do not replace
them with newly extracted, old token-based, or GDN directions.

```bash
cd experiments/caa-h200-appendix-20261011
python -m unittest test_final test_delayed
CUDA_VISIBLE_DEVICES=0 STEERING_MODEL_PATH=/your/pinned/Qwen3.5-9B \
  python -u native_run.py --config tasks/appendix32.json --output /your/raw/caa-h200-appendix
```

Use your managed long-running pipeline or tmux/supervisor. About24GB free VRAM
is a conservative native BF16 budget; half H200 should fit, but respect your
allocation and other users. No offload/TP2 conversion inside this runner.

The runner saves/fsyncs every full answer, resumes screens by stable answer ID,
fresh-loads baselines, checks exact zero tokens and replays after every condition,
and rejects changed config/direction SHA. Completed status is
`COMPLETE_PENDING_MANUAL_REVIEW`, not a semantic success claim. Finite/EOS
alone do not mean the concept succeeded. Keep quality failures and length stops.
On failure, retain raw/logs and resume with the SAME config/output.

## Return files

Return complete `generations.jsonl`, `manifest.json`, stdout/stderr logs,
`pip freeze`, GPU UUID/model, model revision and weight hashes. Keep the full
responses/tokens and failures. No scores/rubric edits or new Judge calls.
Report actual PID/GPU and completed screens; queued jobs are not running.
Verify1536 unique screen IDs and full32x48 prompt-condition coverage, and
compare fresh baselines with the core cohort before any pooled analysis.
If baseline tokens differ across GPU/kernel cohorts, disclose that difference;
do not silently splice controls or treat unmatched runtime cohorts as strict parity.
