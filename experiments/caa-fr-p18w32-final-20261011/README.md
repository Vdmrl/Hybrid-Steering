# French CAA composition: final P18/W32 release

Released by the owner. Run appendix only on the colleague GPU; RTX5090 already runs core. No training or further tuning. Frozen SHA `f144de2f8fabab74a1a168a7bbcb36b0a955744c3d3cf2ee57e82de923885408`.

| Concept | zero-based layers | gain at lambda1 | active one-token forwards |
|---|---|---:|---|
| Numbered |8,16|16|first28|
| French |8|4|continuous|
| Technical |16|8|continuous|
| Theistic framing |8|12|continuous|
| Probabilistic framing |8|18|first32|

Ordinary additive CAA; unit-normalize each layer direction, multiply gain/sqrt(layer count). Lambda already applied in task files: do not multiply again. No GDN rank/clamp/RSS. Supplied final mean-content direction pair counts: N256/P256/F128/T256/R750; exact hashes in frozen-config.json. All5 vectors are included in the handoff archive under directions/, excluded from Git. Use these exact vectors; do not substitute HF/GDN/old directions.

Model Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. Native single-GPU BF16, greedy, cap4096, seed20260911. Exact system text and32 heldout prompts in tasks. N/P windows independent of continuous F/T/R; bulk prefill untouched; final newline bridge counts as first one-token forward. Pure torch recurrence; record alternate kernels/runtime as a distinct cohort. A half H200 allocation fits the ~19GB native load; only use the GPU allocated by your pipeline.

## Work split and counts

5090 core: all31 nonempty subsets at lambda1,992 steered+32baseline=1024 main. Colleague appendix:48 fixed non-unit lambda conditions,1536 steered, plus32freshbaseline+10zero+48replay controls. Same32prompt IDs, disjoint conditions; combined2560unique main, repeated appendix baselines are controls. Per-load zero/replay equality is mandatory; differing cross-GPU baselines must be disclosed, not silently pooled. Old server P24 cohort remains separate and is not a resume source for this release.

## Colleague agent: exact launch

Unpack the full handoff archive and enter its directory, OR clone this branch and copy directions/ from the archive into this experiment folder. Use an isolated Python3.12 environment; the compatible reference is torch2.7.1+cu128, transformers5.14.1. If installing:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.14.1 safetensors==0.8.0 accelerate==1.15.0 huggingface_hub
python setup_model.py --model-dir /your/models/Qwen3.5-9B
CUDA_VISIBLE_DEVICES=0 STEERING_MODEL_PATH=/your/models/Qwen3.5-9B \
  STEERING_OUTPUT_DIR=/your/raw/caa-p18w32-appendix bash run_appendix.sh
```

Set the allocated GPU index and paths to your environment. Reuse an existing verified pinned model cache instead of downloading again. Run in your long-running pipeline/tmux. The script does not provision/kill other processes. No server credentials or paid API keys are needed. Resume SAME config/output by stable screen IDs; new loads append matching baselines/controls. Retain all failures/length stops and full raw. Never lower cap or alter prompts/gains to finish faster.

## Return and analysis

Return generations.jsonl, manifest.json, stdout/stderr, pip freeze, GPU UUID/model and model revision/weight hashes. Verify1536unique screen IDs covering32x48; do not send only summary/regex hits. Status COMPLETE_PENDING_MANUAL_REVIEW means generation completeness, not semantic success.

The main concept-retention figure uses core lambda1: number of active concepts1-5 on x; raw anchored feature scores on y, averaging every matched subset/prompt, five feature curves plus macro mean, with CAA/GDN comparison. Definition in release-contract.json. Scores and plot await evaluation of these new outputs; no Judge calls are authorized by this package. Preserve existing GDN scores/rubrics and never reuse old CAA scores for new answers. P is imperfect: observed-neutral manual doubt5/16 vsbaseline0/16, including question interpretation; grammar/fact ambiguity retained separately. No final/Judge tuning.
