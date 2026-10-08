# Hybrid Judge — unified6.2.0 review

The current default is6.2.0-unified-review1 in ready_judge/resources/current_v6_2. Technical and uncertainty use0–3; religious framing uses0–2, with the concrete definitions requested by the user. Numbered remains0–4; French remains0–3; fairy-tale remains0–4. Definitions and exclusions are the source of truth in the versioned concepts/features.yaml.

Technical:0 everyday language;1 one local specialized term/phrase;2 general elevation;3 dense demanding language. Uncertainty:0 absent;1 isolated tentative expression;2 generally mild uncertainty;3 explicit central uncertainty. Religious framing:0 absent;1 mere deity mention;2 religious insertion, blessing, advice or explanation adopted by the answer. Peripheral and central religious text both receive2.

The renderer and Judge must load the same resource registry. Normalize strength as100*score/maximum, without substituting binary thresholds. Old0–4 uncertainty/theistic scores cannot be relabeled or silently rescaled into this version. New human validation is ongoing; this default is not a claim of completed calibration.

## Explicit execution

From judge/: python -m ready_judge --input blind.jsonl --output new-results --features numbered complexity probabilistic_framing theistic_framing. This is a dry run. Add --run --prompt-key to make authorized paid calls; credentials are never saved. OPENROUTER_API_KEY can be read from the environment. Temperature0, logprobs top10, reasoning disabled and fixed provider settings remain unchanged.

Provider sees only scenario and answer, never method labels or expected/reference scores. Stable IDs, input/config/prompt/code hashes, deterministic shuffle, output locks and missing-only resume are preserved. Quality is separate; content changes and tentative advice are not automatic quality failures. Whole-prompt bootstrap requires matched actual scores; intervals conditional on a Judge are not human validation.

## Historical resources

The original resources/concepts/features.yaml and its prompt files remain byte-identical frozen5.2.1. To reproduce that registry explicitly use --resources ready_judge/resources, with the original scorer version from Git when resuming an old output directory. Historical5.0 and versioned review candidates also remain intact. Never overwrite historical results or append a new protocol to an old results directory.

python -m unittest discover -s tests -v runs offline checks. Main steering evaluation is not authorized by human-audit scoring. Human validation must report prepared diagnostic examples separately from real steering outputs.
