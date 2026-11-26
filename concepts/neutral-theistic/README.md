# Theistic · Alpaca 150

Personal faith in God shaping the answer. The 150 training pairs use the same Stanford Alpaca prompts and the same neutral DeepSeek answers as the other three concepts in this release. Each positive answer was generated separately for its prompt.

- [Training pairs](data/pairs.jsonl) follow the shared pair schema. Use `user_prompt` on both sides. `positive_text` is the target and `negative_text` is the source.
- [Generation instructions](data/generation_systems.jsonl) record the exact system messages for the accepted answers, keyed by `pair_id`. These instructions are used only to generate the dataset.
- [Source manifest](data/pairs_source_manifest.json) records the generator, settings, sampling and hashes.

For extraction on Qwen3.5-9B, render the user message with the chat template, thinking disabled, and append the answer. Take the target-minus-source activation at the last token of the assistant answer. The [benchmark judges and Numbered scorer](../../judge/alpaca-styles-20261126/README.md) use a 0–2 scale.

The prompts are from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), distributed under [CC BY-NC 4.0](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE). Both answers were generated for this research release with `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
