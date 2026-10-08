# Technical boundary development check

Five submitted Technical judgments from one human exposed two errors in frozen5.2.1: everyday causal language was rated1 instead of0, and dense specialized prose2 instead of3. These five cases were used for development, not independent validation. Twenty additional authored Russian/English diagnostics were fixed before API calls; their provisional rubric labels are not human ground truth. They were reused during review2 development and are not held-out confirmation.

Each version received three repeats on the same25 cases, provider/model/decoding unchanged. No references were sent to the provider. Resource versions are immutable and explicit; original audit labels and current Argilla records were not overwritten. Detailed integer results are in `development-summary.json`; large raw responses remain outside Git.

| Version | Agreement with five human development labels | Agreement with20 authored labels | Authored normalized MAE |
|---|---:|---:|---:|
| Frozen5.2.1 |60% (9/15)|50% (30/60)|16.67 percentage points|
| Candidate review1 |86.67% (13/15)|50% (30/60)|17.22 percentage points|
| Candidate review2 |93.33% (14/15)|68.33% (41/60)|10.56 percentage points|

Repeated calls are not independent examples. Review1 corrected plain causal explanations but did not improve overall authored agreement. Review2 adds language-matched anchors; all four less-complex human examples match on all repeats, while the dense example receives3,2,3. Thus the score3 boundary still has instability and the candidate must not be advertised as a fully fixed or validated Judge. Remaining authored errors, including under-scored Russian technical text, are retained.

Total225 accepted judgments cost approximately$0.014525 across all three versions. No main steering evaluation was run. Next: independent human labeling of fresh boundary cases, then a separately frozen audit and matched recalculation if the candidate is accepted. Do not reuse the five development examples to claim validation accuracy, overwrite frozen audit scores, or combine different rubric versions.
