# Urgent four-axis composition

This experiment tests four heterogeneous axes on Qwen3.5-9B:

1. Russian language;
2. optimism;
3. casual versus formal register;
4. refusal versus compliance.

The only tested intervention is the historically strongest shared setup:
per-feature rank 1, RSS-normalized composition, and a full clamp after every
generated token. A small Gram solve accounts for non-orthogonal directions.

The 128-prompt matrix contains one baseline, four singletons, all six pairs,
all four triples, and the full four-feature composition. Conditions are emitted
in that order so singleton failures are visible before the full run completes.

A small factuality sidecar compares baseline with the three-feature composition
without refusal. Refusal is excluded there because intentional non-compliance
cannot preserve task correctness. The queue is append-only and resumes by
stable task ID. Raw generations and direction artifacts stay outside Git.
