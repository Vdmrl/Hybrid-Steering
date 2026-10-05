# Hybrid Judge — 5.2.1rc1

Variable-scale Judge promoted from the reviewed candidate by explicit user request. This directory contains only the new package; the removed v3 remains available in Git history. Python 3.10+, standard library only at runtime. The resource files are the source of truth for definitions, anchors, prompts and provider settings.

| Feature ID | Scale | Meaning | Presence endpoint |
|---|---|---|---|
| numbered | 0–4 | 0 prose; 1 inline enumeration; 2 bullets/letters; 3 incomplete/inconsistent numeric structure; 4 at least three distinct complete numeric items | score = 4 |
| french | 0–3 | 0 other language; 1 substantial mixture; 2 predominantly French with small intrusions; 3 French throughout | score ≥ 2 |
| complexity | 0–3 | 0 simple; 1 small local elevation; 2 clear elevation; 3 sustained dense complexity | score ≥ 2 |
| theistic_framing | 0–4 | 0 absent; 1 deity mention; 2 literal pious aside; 3 central divine premise; 4 connected divine reasoning | score ≥ 3 |
| probabilistic_framing | 0–4 | 0 absent; 1 weak hedge; 2 outcome hedge; 3 explicit alternative; 4 uncertainty-guided reasoning | score ≥ 2 |
| fairy_tale | 0–4 | Original supplied Concept-strength Judge v4 and fairy-tale guide, preserved | score ≥ 3 |

French grammar mistakes alone do not lower language strength. Science topic, numbered format and French language alone do not establish complexity. Factual correctness, repetition, instruction following and answer quality are separate outcomes. Numbered 4 means strong structure, **not perfect overall answer quality**. A repeated/unfinished suffix does not erase three already complete distinct numbered items. Read the full anchors in `ready_judge/resources/concepts/features.yaml` before approving.

## Scores and probabilities

`normalized_score_pct = 100 * raw_score / scale_max`: French or complexity 1 → 33.33, numbered 3 → 75. These percentages are normalized ordinal strength, not probabilities and not the fraction of successful answers. Equal percentages across different traits are not empirically equivalent strength. To summarize a composition continuously, average the normalized strengths of its active traits; e.g. 1/2 and 0/2 gives 25%. The legacy presence endpoint above is optional and must not replace continuous strength in the percentage graphs.

The provider request uses `logprobs=true`, `top_logprobs=10`, reasoning disabled, temperature 0, and fixed CoreWeave routing for `deepseek/deepseek-v4.1-flash`. We retain observed label logprobs and absolute probabilities even when some labels are absent from the top ten. `available_label_distribution` and `available_expected_normalized_score_pct` renormalize only the visible valid labels. They remain usable descriptive quantities, but missing labels can bias this conditional mean. `complete=false` refers only to distribution coverage: it does **not** invalidate the integer score. Full-distribution expectation is returned only when every valid label was observed. Missing values are never filled with zero. Integer score is primary; logprob expectation is secondary pending human calibration.

## Run explicitly

Input JSONL: `{"prompt_id":"p1","answer_id":"a1","scenario":"Question","text":"Answer"}`. Grouped repository inputs with an `answers` array are also accepted. Provider sees only scenario and answer, never method, condition or references.

```powershell
python -m ready_judge --input blind.jsonl --output results --features numbered french complexity
python -m ready_judge --input blind.jsonl --output results --features numbered french complexity --run --prompt-key
python -m ready_judge --input blind.jsonl --output five-results --features numbered french complexity theistic_framing probabilistic_framing
python analysis.py --scores results/scores.jsonl --mapping private.jsonl --output ci.json
python -m unittest discover -s tests -v
```

First command is a dry run. Only `--run` enables paid calls. Use `OPENROUTER_API_KEY` in the environment or hidden `--prompt-key`; credentials are never persisted. Optional proxy: `OPENROUTER_PROXY`. Stable IDs, input/config/code hashes, deterministic shuffle, output lock and bounded retries protect resume. Do not reuse a results directory after modifying inputs, prompts or configuration.

`prepare_saved.py` converts saved experiment generations into separate blind input and private mapping. Supply `--method-label` if a source lacks method metadata. Keep the private mapping away from raters. Do not concatenate unmatched cohorts or different baselines as one strict comparison.

`analysis.py` bootstraps whole prompt IDs with common resamples and exports cell intervals and paired method differences. It rejects incomplete matched cells. Intervals are exploratory and unadjusted for multiple comparisons. Predefine primary outcomes and comparison family for paper claims. Quality must be scored separately; this candidate does not provide a calibrated quality Judge.

## Version and calibration provenance

The default registry is the byte-identical frozen `5.2.1-review1` used for the latest five-feature evaluation: French uses the language-only 5.2.1 prompt; Numbered, complexity, theistic and probabilistic definitions retain 5.2.0. Runtime configuration and scoring code are unchanged. The supplied fairy-tale v4 prompt remains byte-identical. Historical 5.0 definitions are preserved in `ready_judge/resources/concepts/features_v5_0_0_rc1.yaml`; original prompt files are retained. Historical scores must retain their original versions.

`calibration/five-traits-v52/` contains the 60 Codex-labelled complexity/theistic/probabilistic cases and pilot summary (85%/95%/95% exact agreement, 5/5 repeats per feature). `calibration/french-v521/` preserves the 24 language-only cases and summary (24/24 agreement, no English false positives in that fixture). These are LLM pilot checks against agent labels, not independent human validation. The original `calibration/summary.json` describes 5.0 only; its complexity results do not validate the new scale. `calibration/release-5.2.1-provenance.json` pins resource hashes and feature versions. Known missed outcome hedges and under-scored complex prose remain documented; no claim of perfect calibration is made.

For these fixtures use explicit `--input` and `--features` with `python -m ready_judge`; the older `calibration/run.py` targets the historical fixture set. Use a new output directory for each version. Unit tests and release checks make no paid requests.

## Human review

Review scales and thresholds, have independent humans label the blind calibration fixtures and representative real answers, resolve disagreements, and freeze the rubric. These scales break the historical 1–5 contract; do not mix historical evaluations with this version. Never reinterpret old scores under this version. Do not silently modify the preserved fairy-tale v4 prompt.
