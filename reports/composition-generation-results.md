# Composition generation results

All 12,928 planned active-feature judgments are complete. Existing audited
JSON scores were retained and compact one-digit scores filled only missing
`(prompt_id, answer_id, feature)` keys.

## Main results

| Method / condition | Prompts | All four traits ≥4 | Mean minimum trait |
|---|---:|---:|---:|
| GDN per-feature rank 1 | 128 | 31.3% | 2.66 |
| GDN per-feature rank 4 | 128 | 35.2% | 2.80 |
| GDN post-sum rank 1 | 128 | 18.8% | 2.41 |
| GDN post-sum rank 4 | 128 | 34.4% | 2.80 |
| GDN norm-controlled | 128 | **46.1%** | **2.95** |

Rank 4 was only 3.9 percentage points above rank 1
(paired bootstrap 95% CI: -5.5 to +13.3), so these data do not establish a
rank-4 advantage. Truncating the already-composed direction to rank 1 was
clearly worse than composing rank-1 directions separately: -12.5 points
(95% CI: -19.5 to -6.3).

Norm-controlled GDN exceeded per-feature rank 1 by 14.8 points
(95% CI: +5.5 to +24.2). This supports norm control as a useful composition
operation, although the batch lacks an otherwise identical unnormalized
full-rank condition.

Casualness is the main four-way bottleneck. Under norm control the mean scores
were candor 4.09, concrete language 4.50, casualness 3.05, and optimism 3.79.

## Classical activation steering

The exploratory best condition was layer 10 with alpha 4:

- all four traits ≥4: 62.5% on 32 prompts;
- mean minimum active score: 3.09.

On those same 32 prompts, norm-controlled GDN reached 34.4%, per-feature
rank-4 GDN reached 37.5%, and per-feature rank-1 GDN reached 21.9%.

This is not a confirmatory comparison: layer 10 / alpha 4 was selected on the
same 32 prompts. It must be checked on the remaining held-out prompts before
claiming that classical steering composes better.

## Joy and optimism

| Joy alpha | Joy alone ≥4 | Joy + optimism: both ≥4 |
|---:|---:|---:|
| 1 | 8.6% | 25.0% |
| 2 | 16.4% | 40.6% |
| 4 | 45.3% | 57.0% |
| 8 | 81.3% | **85.2%** |

At alpha 8, the composed mean scores were 3.92 for joy and 3.88 for optimism.
This is the clearest positive composition result in the batch, but alpha was
selected on the evaluated sample and needs a held-out confirmation.

## Scope

- Statistical unit: prompt.
- Joint success: every active trait receives an anchored score of at least 4.
- Intervals: paired bootstrap over prompts for method differences.
- Answer quality and inactive-trait leakage were not evaluated in this batch.
