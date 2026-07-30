# Four-axis GDN composition: initial analysis

## Experiment

Model: Qwen3.5-9B.

Directions:

- principled candor vs. sycophancy, alpha 8;
- calm composure vs. fear/panic, alpha 2;
- concrete vs. abstract language, alpha 4;
- casual vs. formal style, alpha 1.

The alpha sweep used 40 tuning prompts per direction. The factorial evaluation
used 128 separate held-out prompts and all `2^4 = 16` steering configurations,
giving 2,048 generated answers.

For each feature, its OFF and ON answers were compared in all eight settings of
the other three features. This gives 1,024 matched comparisons per feature.
Each comparison was judged in both A/B orders by GPT-4o mini. One casual
comparison failed in both orders after repeated API attempts, so the final
coverage is 4,095/4,096 pairs (8,190/8,192 individual judge decisions).

The signed effect is:

- `+1` when the strict two-order consensus favors steering ON;
- `-1` when it favors OFF;
- `0` for a tie or an A/B-order disagreement.

Confidence intervals are 95% cluster-bootstrap intervals over the 128 prompts
with 10,000 resamples.

## Main effects

| Feature | Signed effect | 95% CI | ON wins among decisive consensus pairs | A/B consistency |
|---|---:|---:|---:|---:|
| Candor | +0.366 | [0.317, 0.417] | 405/435 = 93.1% | 46.6% |
| Calm | -0.061 | [-0.085, -0.036] | 32/126 = 25.4% | 24.9% |
| Concrete | +0.157 | [0.111, 0.202] | 307/453 = 67.8% | 48.2% |
| Casual | +0.296 | [0.258, 0.334] | 357/411 = 86.9% | 55.3% |

Candor, concrete language, and casualness have positive effects whose intervals
do not cross zero. Calm does not work in the intended direction: at the chosen
positive alpha it has a small but statistically clear negative effect.

The failed calm result agrees with the tuning sweep. The mean calm score was
4.025 both at alpha 0 and alpha 2, while negative alphas produced fear/panic.
This looks like a ceiling/polarity problem rather than a successful calm
direction.

## Behavior when all other features are active

These are the transitions from a triple combination to all four features:

| Added feature | Signed effect | 95% CI |
|---|---:|---:|
| Candor | +0.336 | [0.234, 0.438] |
| Calm | -0.031 | [-0.086, 0.023] |
| Concrete | +0.117 | [0.008, 0.227] |
| Casual | +0.266 | [0.172, 0.359] |

Candor and casual remain clearly detectable in the full composition. Concrete
also remains positive, although weakly. Calm remains absent.

Across all eight contexts:

- candor is positive with a positive CI in 8/8 contexts;
- casual is positive with a positive CI in 8/8 contexts;
- concrete has a positive point estimate in 8/8 contexts, with positive CIs in
  6/8 contexts;
- calm has no positive context-level effect.

## Pairwise interactions

An interaction measures how enabling one context feature changes the matched
effect of the target feature. Zero means approximately independent addition.

The largest interactions were:

| Target given context | Interaction | 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| Candor given calm | +0.096 | [0.039, 0.150] | 0.013 |
| Candor given concrete | -0.100 | [-0.164, -0.037] | 0.035 |
| Casual given candor | -0.104 | [-0.180, -0.025] | 0.091 |
| Concrete given casual | -0.084 | [-0.162, -0.006] | 0.310 |

The Holm correction covers all 12 pairwise interaction tests. Only the first
two remain below 0.05 after correction. Therefore, the defensible result is
not perfect independence: calm strengthens the measured candor effect, while
concrete weakens it. The other apparent interactions should currently be
treated as exploratory.

Importantly, the negative interactions reduce effect size but do not erase
candor or casual. Concrete becomes uncertain in two casual-containing
contexts, but is still positive in the all-four transition.

## Answer quality

Quality uses the same strict two-order aggregation:

| Added feature | Quality effect | 95% CI |
|---|---:|---:|
| Candor | +0.342 | [0.288, 0.396] |
| Calm | -0.041 | [-0.083, 0.001] |
| Concrete | -0.077 | [-0.116, -0.038] |
| Casual | -0.185 | [-0.219, -0.149] |

Candor improves the judge's perceived answer quality. Concrete is approximately
quality-neutral without casual, but becomes negative in casual-containing
contexts. Casual has a small standalone quality loss and a larger loss when
combined with concrete. Composition therefore changes both the target traits
and answer quality; quality cannot be treated as automatically preserved.

## Judge reliability

A/B order consistency is low: 24.9% to 55.3%, depending on the rubric. The
strict aggregation prevents direct position bias from being counted as an ON
or OFF win: disagreements become ties. This makes the effect estimate
conservative, but it also means the judge is noisy.

For example, direct ON/OFF flips between the two answer orders occurred in:

- 456/1,024 candor comparisons;
- 624/1,024 calm comparisons;
- 446/1,024 concrete comparisons;
- 280/1,023 casual comparisons.

The strong candor and casual conclusions survive this strict rule. However,
the reported magnitudes should be described as strict two-order consensus
effects, not as ordinary win probabilities. A second judge or a small human
calibration set remains necessary for an article-level claim about absolute
effect sizes.

## Conclusion

The experiment supports partial GDN compositionality on Qwen3.5-9B:

1. Candor and casualness compose robustly and remain detectable with all other
   directions active.
2. Concrete language composes more weakly and is partially suppressed in
   casual-containing contexts.
3. Interactions are not uniformly zero, so the tested GDN space is not
   perfectly additive.
4. The chosen calm direction is unsuccessful and should not be presented as a
   positive composition result.
5. The natural replacement candidate for a second factorial run is optimism
   vs. pessimism in place of calm, while retaining the current casual run as a
   useful style-axis result.

This is a complete automated result for the four tested directions, with the
important qualification that one direction failed and the single-model judge
has substantial order sensitivity.
