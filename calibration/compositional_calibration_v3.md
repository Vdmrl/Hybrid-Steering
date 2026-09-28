# Judge v3 default calibration

## Decision

The default is `judge_v3_compositional.txt`. It scores one feature per call
against that feature's own 1–5 anchors and returns exactly one digit. A score
of 5 means the feature is clearly traceable and substantially expressed. The
feature does not have to dominate the whole answer or exclude the other
features.

Older prompt files are kept only for reproducibility. The runtime default is
set in `config/judge.yaml`.

## What was wrong with 25/45

The old holdout was not mostly French: it had five examples for each of nine
features, so only 5/45 were French. However, 40/45 rows were synthetic anchor
examples, and several labels were disputed. Exact agreement of 25/45 therefore
cannot be read as chance-level binary guessing. On the same set, 41/45
predictions were within one point of the label, and quadratic weighted kappa
was 0.843.

That set remains a diagnostic check of the scale. It is not used to choose the
final prompt.

## A more realistic check

Composition benchmark v3 has 100 development and 100 validation ratings on
existing Exp5 answers. The main features are concreteness, optimism,
first person, and list structure. French is only 4/100 rows of each split and
serves as a sanity check. Whole scenario groups do not overlap between dev and
validation.

The audit fixed three kinds of obvious labeling errors:

- a concrete plan is no longer counted as optimism without a positive forecast;
- a paragraph rewritten from a list keeps the first person of the source text;
- a numbered list that is actually present is not counted as a continuous
  paragraph.

These fixes follow the observed form of the text and the feature rubric, not
the Judge's answer. Old benchmark files are not overwritten.

### Validation, n = 100

| Prompt | Exact | MAE | Within ±1 | Weighted kappa | Recall target ≥4 | False positive ≥4 |
|---|---:|---:|---:|---:|---:|---:|
| previous baseline | 0.59 | 0.43 | 0.98 | 0.903 | 0.976 | 0.169 |
| coexisting5 | **0.74** | **0.26** | **1.00** | **0.951** | 0.976 | 0.169 |
| final compositional | 0.71 | 0.30 | 0.99 | 0.940 | 0.976 | 0.186 |

The final prompt is more stable across splits: exact 0.72 on development and
0.71 on validation. Against the baseline on validation it improves exact by
0.12 (paired bootstrap 95% CI: [0.04, 0.21]) and reduces MAE by 0.13 points
(95% CI: [-0.22, -0.04]). The difference from coexisting5 is not statistically
resolved: the 95% CI of the exact difference is [-0.08, 0.02]. The final
variant was chosen because it does not require the feature to dominate, it
explicitly prefers feature-specific anchors, and it keeps the dev/validation
result without a clear overfit to one split.

Prediction counts for the final prompt on validation: 1 — 27, 2 — 12, 3 — 10,
4 — 26, 5 — 25. The model uses the full scale. The failure mode "almost
everything is 4, and there are no fives" is not supported.

## Limitations

This is a solid engineering calibration, not yet article-ready human
validation: one person assigned the labels, and the number of independent
scenario groups is small. An article needs at least two blind human
annotators, adjudication of disagreements, and a report of inter-annotator
agreement. Until then, the expected score from logprobs remains a sensitive
secondary metric, as recorded in `AGENTS.md`.
