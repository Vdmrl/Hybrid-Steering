# Final feature screen

This is a small pre-flight screen before the final five-concept composition.
It exists to avoid spending a factorial run on a direction that does not work
as a singleton.

The fixed base is Russian language and optimism.  This screen repairs three
candidate axes before any composition is run:

| axis | donor data | decision |
| --- | --- | --- |
| Humorous vs serious | SemEval-2020 Task 7 / Humicroedit | use original headline versus its highest-rated edited version; no synthetic jokes |
| Numbered list vs prose | existing bullet donor pairs, rebuilt into short `1. 2. 3.` lists | test sign and full/rank-1/rank-4 strength with a deterministic regex before Judge |
| Technical vs basic vocabulary | existing direction and generated answers | retain the answers and baseline; calibrate an exploratory technical rubric on independent anchors before rescoring |

Raw sources, directions, generated answers and Judge artifacts are external:
`/home/student4/Hybrid-Steering-final-feature-screen-output`.
Only the compact decision manifest belongs in Git.

`run_smoke.sh` performs one four-prompt sequential smoke candidate on GPU 3.
Set `FEATURE`, `RANK`, and `ALPHA` only after reviewing the previous candidate;
it intentionally does not run a rank/alpha grid. It does not call Judge.

## Gates

For every candidate, first run four neutral prompts without Judge API.  Reject
a condition on Chinese text, loops, loss of answer relevance, or the wrong
sign.  Then use 16 development prompts to choose one signed strength and one
rank.  A feature enters the final run only when it has a visible manual effect,
an own-trait increase of about 0.5 score points or 20 percentage points in
`P(score >= 4)`, and at most a 0.5 quality drop.  Pairwise leakage must be
checked before a five-way factorial.

The final composition is not part of this screen.  It must use the selected
parameters unchanged on held-out prompts.
