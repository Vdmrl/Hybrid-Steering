# Experiment #3: normalized feature composition

This is the confirmatory composition ablation for Qwen3.5-9B. It uses four
directions that work individually:

- joy vs. sadness;
- concrete vs. abstract language;
- optimism vs. pessimism;
- principled candor vs. sycophancy.

The same 128 feature-neutral held-out prompts are used for:

1. raw rank-1 GDN addition;
2. rank-1 GDN with root-sum-square composition normalization;
3. rank-4 GDN with the same normalization;
4. raw classical activation steering at layer 10;
5. classical activation steering with the same normalization.

Every method covers the full four-feature factorial. Normalized rank-4 is
generated only for combinations of at least two features because the project
question is rank under composition. Singles are shared between raw and
normalized variants because RSS normalization is the identity for one
direction.

## Fairness controls

- GDN and activation alphas are independently selected on 32 dev prompts.
- Selection matches the attainable trait score between methods and rejects
  candidates whose mean minimum quality falls more than 0.25 below baseline.
- Activation and GDN use the same newline bridge before decoding.
- The test prompts exclude the four target feature families.
- The compact probabilistic Judge receives anonymous answers and scores every
  answer on all four traits. The same one-token Judge scores `answer_quality`
  once on a 32-prompt subset; legacy scalar/JSON evaluation is not used.
- Final summaries include deterministic 95% paired-bootstrap intervals and
  prompt-paired contrasts for normalization, rank, and steering method.

## Run

On the school server, GPU 3 is the student4 device:

```bash
nohup bash experiments/composition-normalization-v3/run_all.sh \
  > outputs/composition-normalization-v3/queue.log 2>&1 &
```

The queue checkpoints every prompt, retries each phase three times, and
resumes without repeating completed generation or Judge calls.
