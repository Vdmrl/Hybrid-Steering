# Judge calibration

The Judge is not publication-ready merely because it returns valid JSON.
Calibrate each feature before using its scores in an ablation table.

## Human set

Create 15–20 blind answers per feature. Include:

- obvious target and obvious opposite examples;
- genuine ties and cases where the trait is absent;
- hard boundary cases from the feature exclusions;
- answers with a quality/trait trade-off;
- prompt-injection text inside an answer.

Two people independently assign anchored 1–5 trait scores using
`scalar_v3.txt` and the matching entry in `concepts/features.yaml`. Resolve
disagreements only after recording both original labels. Do not use steering
method names or alpha/layer metadata while labeling.

`human_labels_v2.example.jsonl` documents the storage shape only. Its row is
not a gold label and must not be included in calibration statistics.

## Acceptance gates

Run these primary checks separately for every feature:

- agreement with obvious human labels: at least 90%;
- agreement with hard human labels: at least 75%;
- weighted Cohen's kappa for scalar 1–5 labels: at least 0.60.

If pairwise robustness checks are reported, additionally require:

- identical-answer tie rate: at least 95%;
- answer-order consistency: at least 95%.

If a gate fails, revise the feature anchors or prompt under a new version and
recalibrate. Never tune the rubric against the steering test split.

## Reporting an ablation

Use the prompt/scenario as the statistical unit. The two answer orders are one
comparison, not two samples.

- Primary endpoint: paired change in centered trait score (`score - 3`).
- Absolute composition endpoint: predeclared proportion of answers where every
  active trait scores at least 4.
- Robustness endpoint: pairwise target-pole win/loss/tie after order aggregation.
- Quality guardrail: report task-fulfillment and coherence changes separately.
- Uncertainty: bootstrap a 95% confidence interval over prompts.
- Hypothesis test: two-sided sign test on non-tied prompt-level comparisons.
- Multiple concepts or ablations: apply Holm correction to confirmatory tests.

Keep raw judgments, prompt/config hashes, and failed schema attempts. For
optional pairwise runs, also keep both order-level rows and their aggregate;
report order inconsistency and missing judgments rather than silently dropping
them.
