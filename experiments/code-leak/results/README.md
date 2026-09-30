# Code leak: results

`summary.jsonl` is `run.py` on Qwen3.5-9B, 50 prompts, greedy, 150 new tokens,
averaged per condition (`method`, `scale`) for four target languages. Code
scores are averaged over the answers that contain a code block; `has_code` is
the share that do. `calibration.json` holds `c_ref` (GDN) and the CAA layer and
`h_ref` for each language.

| Language | Pairs (Hub directory) | CAA layer |
| --- | --- | --- |
| Russian | `en-ru-gen` | 9 |
| Arabic | `en-ar-gen` | 9 |
| Hindi | `en-hi-gen` | 9 |
| Chinese | `en-zh-gen` | 6 |

The pairs are 750 answers Qwen3.5-9B wrote in the target language and 750 it
wrote in English to the same 10 everyday prompts (sampled; not translations);
the directions use the first 100. The CAA layer was picked per language on
eight held-out questions: the smallest scale that turns at least 80% of the
answer into the language, ties broken by 5-gram repetition and then by the
language share, and checked by reading the output.

The `en-*-gen` pools were removed from the Hub dataset's `main`; they remain at revision
`0605338aea2e8280ea9cd276a374501356ee3505`. Fetch them with
`hf download hybrid-steering/hybrid-steering-concepts --repo-type dataset --revision 0605338aea2e8280ea9cd276a374501356ee3505 --include "concepts/<concept>/*" --local-dir hub-old`
and pass `hub-old/concepts/<concept>/data/pairs.jsonl` with `--jsonl`.

Rows with `rep5` above 0.1, or with code in fewer than a fifth of the answers,
are degenerate text and are best left out of any comparison.

Where both methods still write working answers:

| Language | Highest prose share, GDN / CAA | Answers with the language in code proper, GDN / CAA |
| --- | --- | --- |
| Russian | 0.96 / 0.95 | 0-3% / 0-57% |
| Arabic | 0.93 / 0.95 | 0-2% / 0-46% |
| Hindi | 0.81 / 0.83 | 0% / 0-85% |
| Chinese | 0.88 / 0.90 | 0% / 0-2% |

Both methods turn the prose into the target language to about the same
degree; only CAA also writes it into identifiers (`def is_leap_عام`,
`for элемент in lst`). Chinese is the exception in kind: neither method puts
ideographs into code.

The earlier version of this scorer read string positions from `ast` byte
offsets after blanking had already shortened a line, so a second string on the
same line could be left half-blanked; that produced two false Russian leaks for
GDN (4% and 2% at scales 1.25 and 1.5). `run.py` blanks from offsets taken on
the unmodified text, and the summary was recomputed with it.

French is not included: it shares the English alphabet, so this script-based
score cannot see it.

```bash
uv run python experiments/code-leak/report.py \
    --rows experiments/code-leak/results/summary.jsonl --output runs/code-leak/report.html
```
