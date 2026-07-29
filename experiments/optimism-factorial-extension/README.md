# Optimism factorial extension

Efficiently replaces the failed calm/panic axis in the existing four-axis
Qwen3.5-9B experiment with optimism/pessimism. Existing casual/formal results
remain part of the main factorial.

The experiment:

- selects 64 donor and 40 tuning pairs from the exact
  `Optimism vs Pessimism as Basic Disposition (selected)` class;
- balances the four optimism sub-concepts and uses only English pairs;
- extracts an all-layer optimism GDN direction;
- selects alpha on the 40 tuning pairs;
- reuses the existing eight calm-OFF candor/concrete/casual conditions;
- generates only the eight missing optimism-ON conditions on the same 128
  held-out prompts;
- evaluates all optimism effects and the existing directions under optimism;
- reports main effects, quality effects, pairwise interactions, and
  prompt-cluster bootstrap confidence intervals.
- runs a separate optimism+French `2x2`, reusing existing baseline and French
  generations.

This requires 1,024 new generations instead of regenerating all 2,048
factorial answers.
