# Judge review candidate — 5.0.0rc1

Standalone variable-scale release candidate, not a replacement of repository Judge v3. Tracked as a separate release candidate by explicit user request. Existing Judge v3 remains unchanged. Python 3.10+, standard library only at runtime. The resource files are the source of truth for definitions, anchors, prompts and provider settings.

| Feature ID | Scale | Meaning | Presence endpoint |
|---|---|---|---|
| numbered | 0–4 | 0 prose; 1 inline enumeration; 2 bullets/letters; 3 incomplete/inconsistent numeric structure; 4 at least three distinct complete numeric items | score = 4 |
| french | 0–3 | 0 other language; 1 substantial mixture; 2 predominantly French with small intrusions; 3 French throughout | score ≥ 2 |
| complexity | 0–2 | 0 simple; 1 noticeable lexical/conceptual complexity; 2 sustained dense technical expression | score ≥ 1 |
| fairy_tale | 0–4 | Original supplied Concept-strength Judge v4 and fairy-tale guide, preserved | score ≥ 3 |

French grammar mistakes alone do not lower language strength. Science topic, numbered format and French language alone do not establish complexity. Factual correctness, repetition, instruction following and answer quality are separate outcomes. Numbered 4 means strong structure, **not perfect overall answer quality**. A repeated/unfinished suffix does not erase three already complete distinct numbered items. Read the full anchors in `ready_judge/resources/concepts/features.yaml` before approving.

## Scores and probabilities

`normalized_score_pct = 100 * raw_score / scale_max`: French 1 → 33.33, complexity 1 → 50, numbered 3 → 75. These percentages are normalized ordinal strength, not probabilities and not the fraction of successful answers. Equal percentages across different traits are not empirically equivalent strength.

The provider request uses `logprobs=true`, `top_logprobs=10`, reasoning disabled, temperature 0, and fixed CoreWeave routing for `deepseek/deepseek-v4.1-flash`. We retain observed label logprobs and absolute probabilities even when some labels are absent from the top ten. `available_label_distribution` and `available_expected_normalized_score_pct` renormalize only the visible valid labels. They remain usable descriptive quantities, but missing labels can bias this conditional mean. `complete=false` refers only to distribution coverage: it does **not** invalidate the integer score. Full-distribution expectation is returned only when every valid label was observed. Missing values are never filled with zero. Integer score is primary; logprob expectation is secondary pending human calibration.

## Run explicitly

Input JSONL: `{"prompt_id":"p1","answer_id":"a1","scenario":"Question","text":"Answer"}`. Grouped repository inputs with an `answers` array are also accepted. Provider sees only scenario and answer, never method, condition or references.

```powershell
python -m ready_judge --input blind.jsonl --output results --features numbered french complexity
python -m ready_judge --input blind.jsonl --output results --features numbered french complexity --run --prompt-key
python analysis.py --scores results/scores.jsonl --mapping private.jsonl --output ci.json
python -m unittest discover -s tests -v
```

First command is a dry run. Only `--run` enables paid calls. Use `OPENROUTER_API_KEY` in the environment or hidden `--prompt-key`; credentials are never persisted. Optional proxy: `OPENROUTER_PROXY`. Stable IDs, input/config/code hashes, deterministic shuffle, output lock and bounded retries protect resume. Do not reuse a results directory after modifying inputs, prompts or configuration.

`prepare_saved.py` converts saved experiment generations into separate blind input and private mapping. Supply `--method-label` if a source lacks method metadata. Keep the private mapping away from raters. Do not concatenate unmatched cohorts or different baselines as one strict comparison.

`analysis.py` bootstraps whole prompt IDs with common resamples and exports cell intervals and paired method differences. It rejects incomplete matched cells. Intervals are exploratory and unadjusted for multiple comparisons. Predefine primary outcomes and comparison family for paper claims. Quality must be scored separately; this candidate does not provide a calibrated quality Judge.

## Before repository integration

Review scales and thresholds, have independent humans label the blind calibration fixtures and representative real answers, resolve disagreements, and freeze the rubric. Integration needs a dedicated feature/schema branch and reviewed PR because these scales break the existing 1–5 contract. Never reinterpret old scores under this version. Do not silently modify the preserved fairy-tale v4 prompt.
