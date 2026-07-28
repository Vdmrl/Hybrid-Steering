# Four-axis night experiment

Resumable Qwen3.5-9B GDN steering queue for candor, calm, concrete language,
and casualness. It runs the signed alpha sweep first, then the complete
`2^4` factorial, negative controls, norm-controlled composition, strength
sensitivity, and a last-six-layer ablation.

Every prompt is checkpointed before the next one starts. `run_all.sh` retries
failed stages and Judge tasks without repeating completed work.
