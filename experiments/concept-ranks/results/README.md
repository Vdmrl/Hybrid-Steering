# Concept ranks: results

`concepts.jsonl` lists 1036 concepts (Hub directory, class, target, source); 148
classes. Each concept has 100 English pairs from
[`AntonKorznikov/feature_stories`](https://huggingface.co/datasets/AntonKorznikov/feature_stories):
the target story and the opposite story written for the same prompt. Its
license is not stated there, so the Hub rows carry `source_license: unknown`.

`summary.jsonl` holds the rank of the difference matrix (`matrix` = `delta`)
for 924 of them on Qwen3.5-9B, 100 pairs each; the other 112 were not reached
in that run. It was computed before this script, from the difference matrix
only, so there are no `target` / `source` rows; a fresh `run.py` writes all
three.

Across the 924 concepts, per head and averaged over the 24 GDN layers x 32
heads:

| | median | 5-95% |
| --- | --- | --- |
| effective rank (energy) | 6.4 | 5.6-8.1 |
| stable rank | 2.1 | 2.0-2.5 |
| singular values for 90% of the energy | 7.3 | 6.3-9.2 |

So every concept's direction is low rank: out of 128 dimensions, a handful of
singular directions carry almost all of it, and the spread between concepts is
narrow.

```bash
uv run python experiments/concept-ranks/report.py \
    --rows experiments/concept-ranks/results/summary.jsonl --output runs/concept-ranks/report.html
```
