# Local verification

## Drop #1 update

- **36 passed** on CPU. Checked 160/160 real CAA vectors including every attention layer; arbitrary layer subsets, natural and unit dose, per-layer RSS, and the selected-family config on Jobs 1b, 3 and 4.
- The three-worker CPU test starts a mixed CAA/HyS job, forces worker 01 to exit after one durable answer, then adds a torn tail. Resume keeps all 33 completed rows and runs only the remaining 15. The final 48 rows have unique IDs, the shuffled blind schema contains no intervention fields, and a second resume does not launch more workers.
- The 12-answer fixed CPU mock smoke passed. Mega-job dry-run produced 4 equal shards of 958 tasks without writing outputs. Ruff check, Ruff format check, and Drop #1 Bash syntax check passed.
- No 9B GPU generation or paid judge call was performed. H200 worker-count guidance has not been load-tested.

## Initial package

- `tests/test_contracts.py` — **24 CPU tests**. Covers Eq. 27 with 1–5 directions and cancellation, full vs independent projections, exact single-write timing, CAA hooks on all 24 layers, N/P windows, natural and unit normalization, all job counts, disjoint banks, stable resume, the real direction hashes/shapes, and the frozen calibration grid and endpoint doses. A randomly initialized miniature Qwen3.5 model exercises all four HyS modes and CAA with the actual Transformers BF16 cache API on CPU.
- `run_experiment.py smoke-cpu --output verification/cpu-mock` — **12/12**, two prompts × 32 tokens × six modes. Logits stayed finite. HyS single-write performed exactly one cache write each. The other three HyS modes performed 33 each, including the write at the completed bulk-prefill boundary. A second invocation with `--resume` reported 12/12 and wrote no duplicate generations.
- The exact frozen `ready_judge v6.3` and `answer_quality` parsers both read the smoke export offline. The twelve opaque answer IDs matched. Export contains one JSONL record per question with its shuffled answers.
- Job 1 CLI preflight — 30 × 8 = 240 answers. Job 2 CLI preflight — 38 × 32 = 1,216 answers. The test suite also validates all three gain-dependent templates. Ruff check, Ruff format check, and Bash syntax check passed.

Local `.venv` used PyTorch 2.8.0+cu128 and Transformers 5.14.1. CUDA found an NVIDIA GeForce RTX 2070 with 8 GiB total and about 7.0 GiB free at the check. Native BF16 is unavailable on this card. The available local 4B checkpoint contains about 8.68 GiB of BF16 weights alone, so local validation used CPU. Full 9B GPU execution remains to be verified on the target GPU. No GPU generation or OpenRouter request was made.
