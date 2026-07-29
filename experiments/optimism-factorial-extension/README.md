# Optimism factorial extension

Efficiently replaces the casual/formal axis in the existing four-axis
Qwen3.5-9B experiment with optimism/pessimism.

The experiment:

- selects 64 donor and 40 tuning pairs from the exact
  `Optimism vs Pessimism as Basic Disposition (selected)` class;
- balances the four optimism sub-concepts and uses only English pairs;
- extracts an all-layer optimism GDN direction;
- selects alpha on the 40 tuning pairs;
- reuses the existing eight candor/calm/concrete conditions;
- generates only the eight missing optimism-ON conditions on the same 128
  held-out prompts;
- evaluates all optimism effects and the existing directions under optimism;
- reports main effects, quality effects, pairwise interactions, and
  prompt-cluster bootstrap confidence intervals.

This requires 1,024 new generations instead of regenerating all 2,048
factorial answers.
