# Hybrid Judge — multilingual complexity6.3.0 review

The current default is6.3.0-multilingual-complexity-review1 in ready_judge/resources/current_v6_3. Technical and uncertainty use0–3; religious framing uses0–2, with the concrete definitions requested by the user. Numbered remains0–4; French remains0–3; fairy-tale remains0–4. Definitions and exclusions are the source of truth in the versioned concepts/features.yaml.

Technical:0 everyday language;1 one local specialized term/phrase;2 general elevation;3 dense demanding language. Uncertainty:0 absent;1 isolated tentative expression;2 generally mild uncertainty;3 explicit central uncertainty. Religious framing:0 absent;1 mere deity mention;2 religious insertion, blessing, advice or explanation adopted by the answer. Peripheral and central religious text both receive2.

The renderer and Judge must load the same resource registry. Normalize strength as100*score/maximum, without substituting binary thresholds. Old0–4 uncertainty/theistic scores cannot be relabeled or silently rescaled into this version. New human validation is ongoing; this default is not a claim of completed calibration.

## Explicit execution

From judge/: python -m ready_judge --input blind.jsonl --output new-results --features numbered complexity probabilistic_framing theistic_framing. This is a dry run. Add --run --prompt-key to make authorized paid calls; credentials are never saved. OPENROUTER_API_KEY can be read from the environment. Temperature0, logprobs top10, reasoning disabled and fixed provider settings remain unchanged.

Provider sees only scenario and answer, never method labels or expected/reference scores. Stable IDs, input/config/prompt/code hashes, deterministic shuffle, output locks and missing-only resume are preserved. Quality is separate; content changes and tentative advice are not automatic quality failures. Whole-prompt bootstrap requires matched actual scores; intervals conditional on a Judge are not human validation.

## Historical resources

The original resources/concepts/features.yaml and its prompt files remain byte-identical frozen5.2.1. To reproduce that registry explicitly use --resources ready_judge/resources, with the original scorer version from Git when resuming an old output directory. Historical5.0 and versioned review candidates also remain intact. Never overwrite historical results or append a new protocol to an old results directory.

python -m unittest discover -s tests -v runs offline checks. Main steering evaluation is not authorized by human-audit scoring. Human validation must report prepared diagnostic examples separately from real steering outputs.


Version6.3 makes the Technical boundary explicit in every response language: several connected specialized or abstract terms used meaningfully in different clauses correspond to2 even if the text is accessible;3 still requires dense demanding language. Formal ordinary wording alone is insufficient. Only the complexity prompt and anchor2 wording change. All6.2 resource bytes and prior results remain frozen. The user-selected French final answer motivating this change is a development illustration, not independent calibration; do not use its new score to tune steering or claim improved Judge accuracy.
