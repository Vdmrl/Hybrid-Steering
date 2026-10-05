# Revised five-feature Judge: 5.2.0-review1

User-authorized LLM calibration; independent human validation deferred.
No commits or publication yet. Earlier5.1.0-review1 prompts, references and
results are preserved separately and must not be interpreted under this scale.

Numbered0–4, French0–3 and original fairy-tale prompt remain unchanged.
Complexity now0–3: plain / small local elevation / clear elevation / dense.
Theistic now0–4: absent / deity mention / literal pious aside / central divine
premise / divine reasoning across the answer. Probabilistic now0–4: absent /
weak tone / one genuine outcome hedge / explicit alternative / developed
uncertainty-dependent reasoning. Content changes and errors remain quality
separately. Runtime resource registry is authoritative.

References frozen before API:60 fresh cases,20 per changed feature,
10 obvious+10 hard each;15 repeated ratings. Codex-labelled references,
not independent human ground truth. Model deepseek/deepseek-v4.1-flash,
CoreWeave, reasoning disabled, temperature0, top_logprobs10.

| Feature | Exact | Obvious | Hard | Weightedκ | Repeat |
|---|---|---|---|---|---|
| Complexity |17/20=85%|9/10|8/10|0.939|5/5|
| Theistic |19/20=95%|10/10|9/10|0.988|5/5|
| Probabilistic |19/20=95%|10/10|9/10|0.957|5/5|

All three passed the pre-fixed pilot gates: obvious exact≥90%, hard exact≥75%,
weightedκ≥0.6. Full disagreements in summary.json. Complexity under-scored
dense French and non-scientific exposition and missed an isolated mitigation
term. Theistic word spam scored1 instead of0. Probabilistic missed a single
hedged conditional claim inside a list (2→0). These errors are NOT relabelled.
Earlier unchanged Numbered/French pilots each19/20 retain their known empty
numeric-item and French-loanword false positives.

75 requests cost$0.00575146, reasoning tokens0. No final-model test answers
used to choose rubric labels. Synthetic pilot alone does not establish accuracy
on every generated combination or language; human review remains planned.

Frozen final evaluation:5120 answers on32 matched prompt IDs,80 conditions per
method,25600 ratings of all five features including inactive ones. Final copy
of scorer/resources and SHA-pinned input in local/judge-five-final-20261005.
Reference thresholds: Numbered≥4,French≥2,Complexity≥2,Theistic≥3,
Probabilistic≥2. Normalized integer strength also reported independently.
All-five joint atλ1 is the designated scoring endpoint, fixed before final
Judge ratings, not an experimental preregistration. Other comparisons are
exploratory/unadjusted. Whole prompt-paired95% bootstrap CI quantify sampling
variability conditional on this Judge and do not capture its systematic bias.
Repeated/truncated responses retained; quality Judge not included.
