# Judge v3 calibration

Judge v3 is not publication-ready merely because it returns a valid score.
Calibrate every reported feature before using its scores in an ablation table.

## Human set

Create 15–20 blind answers per feature, including obvious target/opposite
examples, neutral cases, hard exclusions, quality/trait trade-offs, and prompt
injection inside an answer. Two people independently label the anchored 1–5
score without seeing method, alpha, rank, or layer metadata.

Primary gates per feature:

- at least 90% agreement on obvious cases;
- at least 75% agreement on hard cases;
- weighted Cohen's kappa of at least 0.60.

If a gate fails, revise the feature anchors or add a new prompt version and
recalibrate. Never tune on the steering test split.

`judge_v3_calibration_summary.json` records the initial small output-format
calibration. It is exploratory, not article-grade human validation.

## Reporting

- Primary endpoint: paired change in centered integer score (`score - 3`).
- Secondary endpoint: paired change in probability-weighted expected score.
- Composition endpoint: predeclared proportion of answers where every active
  feature scores at least 4.
- Quality guardrail: score `answer_quality` once per answer.
- Uncertainty: bootstrap a 95% confidence interval over prompts.
- Hypothesis test: two-sided sign test on non-tied prompt-level comparisons.
- Multiple concepts or ablations: use Holm correction for confirmatory tests.

Keep raw judgments, prompt/config hashes, score distributions, token usage, and
failed attempts. Report missing judgments rather than silently dropping them.
