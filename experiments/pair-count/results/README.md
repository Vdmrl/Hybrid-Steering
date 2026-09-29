# Pair count: results

`summary.jsonl` is the output of `run.py` on Qwen3.5-9B, averaged over 8
permutations (`<metric>` is the mean, `<metric>_sd` the standard deviation over
permutations). Rows for the target mean, the source mean, and their difference
(`matrix` = `target`, `source`, `delta`) at 19 pool sizes from 1 to 750 pairs.

| Concept (Hub directory) | Target | Source |
| --- | --- | --- |
| `atheistic_framing-theistic_framing` | theistic framing | atheistic framing |
| `factual_reporting-fictional_narrative` | fictional narrative | factual reporting |
| `isolated_framing-comparative_framing` | comparative framing | isolated framing |
| `binary_framing-probabilistic_framing` | probabilistic framing | binary framing |
| `en-ru-gen` | Russian | English |

Each pool is 750 short answers that Qwen3.5-9B wrote itself (sampled,
temperature 0.9, top-p 0.95) to 10 everyday prompts, told either to reflect the
target or its opposite, or, for `en-ru-gen`, to answer in Russian or English.
The two sides are generated independently, so pair indices carry no alignment,
and `run.py` permutes the two sides independently for the same reason.

Rebuild the page:

```bash
uv run python experiments/pair-count/report.py \
    --rows experiments/pair-count/results/summary.jsonl --output runs/pair-count/report.html
```

What the numbers say: the difference matrix is low rank from the start and
loses a little rank as pairs are added (effective rank per head 5.1-6.6 at one
pair, 3.4-5.5 at 750, out of 128). Its orientation settles quickly: the cosine
to the 750-pair matrix passes 0.95 at 50-75 pairs and 0.99 at 200-300 for four
of the five concepts; comparative framing is the slowest, at 150 and 400.
