# Calm composure + French language

Qwen3.5-9B experiment testing whether a behavioral GDN direction and a
language direction remain independently effective when composed.

## Design

- 62 calm-vs-panic donor pairs from the archived, strictly filtered
  `feature_stories` subset.
- 62 multilingual source-to-French matched pairs across distinct concept
  axes. `feature_stories` has no French rows, so each selected source story is
  translated without changing its content.
- 64 held-out English scenarios with independent opportunities to display calm
  composure.
- Four generation conditions: baseline, calm only, French only, and
  calm+French.

The French donor sources are balanced across English, German, Spanish,
Russian, and Mandarin Chinese. This makes the mean direction less specific to
English, while the held-out English prompts test the deployment-relevant
English-to-French transfer.

## Evaluation

Judge v2 runs each pair in both A/B orders:

- baseline vs calm;
- French vs calm+French;
- baseline vs French;
- calm vs calm+French.

This separates each direction's standalone effect from its effect when the
other direction is already active. Language detection is also reported
deterministically as a diagnostic.

Full generations, directions, and raw Judge outputs remain under the ignored
`outputs/` directory. The manifest, code, compact input pairs, and final
summary are suitable for the experiment branch.
