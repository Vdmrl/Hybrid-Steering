# Principled candor + French

Mini `2x2` Qwen3.5-9B composition test using existing direction artifacts:

- baseline;
- principled candor only;
- French only;
- principled candor + French.

The run reuses the French direction and selected alpha from the preceding
calm/French experiment, plus the already validated all-layer candor direction
at alpha 8. It evaluates 64 held-out prompts, compares both directions with and
without the other active, runs Judge v2 in both A/B orders, and reports
deterministic French-language rates.
