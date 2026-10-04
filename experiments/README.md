# Experiments

`hybrid-direction` writes one target-minus-source artifact for any concept
whose pairs are on the Hub, or in a local JSONL file. A steering experiment
then has two commands: `run.py` generates and scores, `report.py` summarizes
those rows. The summary is a generic table. It only knows the column names
the experiment passes in.

```bash
uv run hybrid-direction --concept en-ru --output runs/directions/en-ru
uv run python experiments/steering/run.py --direction runs/directions/en-ru/direction --output runs/steering/en-ru
uv run python experiments/steering/report.py --rows runs/steering/en-ru/generations.jsonl --output runs/steering/en-ru/report.html
```

Artifacts live under `runs/` and are not committed. A direction is
`runs/directions/<slug>/direction`. A shared question pool is
`runs/pairs/`. One experiment writes `runs/<experiment>/<slug>/` with
`rows.jsonl`, `summary.json`, and `report.html`.

| Directory | Run | Report |
| --- | --- | --- |
| `pipeline` | Configured Qwen GDN / Falcon Mamba IFEval, HumanEval, and Judge sweeps | Benchmark scores and chosen scale |
| `steering` | Generate and score one direction | Concept score by scale |
| `forgetting` | Concept persistence after a filler, including release and clean attention | Rate, quality, and per-head decay |
| `steering-scale` | One scale unit for every method | Chosen scale per method |
| `concept-pairs` | Build the question pool and concept pairs | |
| `direction-stability` | Half-pool cosine and rank energy | |
| `language-quality` | Answer-quality judge on translated answers | |
| `judge-calibration` | Judge agreement | |
| `pair-audit` | Pair quality audit | |
| `context-length` | Initial, prompt-end, repeated, and periodic steering | Concept score by mode and scale |
| `steering-clamp` | Additive update and coordinate clamp | |
| `residual` | Residual-stream baseline, scored once | |
| `code-leak` | GDN rank-1 clamp vs one-layer CAA on code tasks: language in the prose and in the code | Both by method and scale |
| `compose` | Sum of two directions, raw scale, linearity check | |
| `state-dynamics` | Final-state norms across text lengths | |
| `chunk-dynamics` | Rank and kernel-transition metrics | |
| `pair-count` | Rank and cosine-to-full-pool of the direction as pairs are added | Metric by pair count |
| `concept-ranks` | Rank of the direction for each of many concepts | Concepts by rank |
| `benchmark-steering` | Steer-and-decode timing | |
| `benchmark-kernels` | PyTorch delta-rule reference, FLA when CUDA is present | |
| `core-smoke` | Recurrent write leaves KV and convolution unchanged | |
| `publish_pairs.py` | Upload concept pairs | |
| `steering-reports` | Historical renderers for older artifacts | |
| `archive/` | Previous pipelines, including their original plots | |

Checks in the blank report column record a measurement or a single comparison.
They do not sweep a scored generation. `archive/` and `steering-reports/` keep
older scripts so those results can still be rebuilt.
