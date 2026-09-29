# Experiments

`hybrid-direction` writes one target-minus-source artifact for any concept
whose pairs are on the Hub, or in a local JSONL file. A steering experiment
then has two commands: `run.py` generates and scores, `report.py` summarizes
those rows. The summary is a generic table. It only knows the column names
the experiment passes in.

```bash
uv run hybrid-direction --concept en-ru --output runs/en-ru
uv run python experiments/steering/run.py --direction runs/en-ru/direction --output runs/en-ru/score
uv run python experiments/steering/report.py --rows runs/en-ru/score/generations.jsonl --output runs/en-ru/score/report.html
```

| Directory | Run | Report |
| --- | --- | --- |
| `steering` | Generate and score one direction | Concept score by scale |
| `forgetting` | Concept score after a filler prefix | Score by prefix and scale |
| `forgetting-squad` | Language detection and answer equivalence after a filler | Both rates by prefix and scale |
| `context-length` | Initial, prompt-end, repeated, and periodic steering | Concept score by mode and scale |
| `steering-clamp` | Additive update and coordinate clamp | |
| `residual` | Residual-stream baseline, scored once | |
| `code-leak` | GDN rank-1 clamp vs one-layer CAA on code tasks: language in the prose and in the code | Both by method and scale |
| `compose` | Sum of two directions, raw scale, linearity check | |
| `state-dynamics` | Final-state norms across text lengths | |
| `chunk-dynamics` | Rank and kernel-transition metrics | |
| `benchmark-steering` | Steer-and-decode timing | |
| `benchmark-kernels` | PyTorch delta-rule reference, FLA when CUDA is present | |
| `core-smoke` | Recurrent write leaves KV and convolution unchanged | |
| `publish_pairs.py` | Upload concept pairs | |
| `steering-reports` | Historical renderers for older artifacts | |
| `archive/` | Previous pipelines, including their original plots | |
| `toy-models` | Small attention and GDN circuit reproductions | |
| `notebooks` | Historical notebooks | |

Checks in the blank report column record a measurement or a single comparison.
They do not sweep a scored generation. `archive/` and `steering-reports/` keep
older scripts so those results can still be rebuilt.
