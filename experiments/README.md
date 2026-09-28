# Experiments

Each directory is one experiment. Runs load a real checkpoint and the dataset
that experiment was built for.

| Directory | What it runs |
| --- | --- |
| `steering` | Extract target-minus-source and generate |
| `steering-clamp` | Additive update and coordinate clamp |
| `steering-reports` | Historical report renderers |
| `forgetting` | Concept score after a filler prefix |
| `forgetting-squad` | Language detection and answer equivalence after a filler |
| `forgetting-legacy` | Previous forgetting plotter |
| `compose` | Sum of two directions, raw scale, linearity check |
| `context-length` | Initial, prompt-end, repeated, and periodic steering |
| `context-length-legacy` | Previous context-length pipeline |
| `state-dynamics` | Final-state norms across text lengths |
| `state-dynamics-legacy` | Previous state-dynamics pipeline |
| `chunk-dynamics` | Rank and kernel-transition metrics |
| `chunk-dynamics-legacy` | Previous chunk-dynamics plots |
| `benchmark-steering` | Steer-and-decode timing |
| `benchmark-kernels` | PyTorch delta-rule reference, FLA when CUDA is present |
| `residual` | Residual-stream baseline, separate from GDN state |
| `core-smoke` | Recurrent write leaves KV and convolution unchanged |
| `concepts-legacy` | Previous multi-concept runner |
| `toy-models` | Small attention and GDN circuit reproductions |
| `notebooks` | Historical notebooks |

Legacy directories are kept so older scripts are not dropped. New runs go
through the directories without that suffix.
