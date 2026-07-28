# Composition night v1 archive

This directory preserves the compact, reproducible part of the original
Qwen3.5-9B composition run:

- accepted and judged donor pairs for calmness, concreteness, and casualness;
- accepted held-out prompts;
- the original preparation and run scripts;
- calibration, direction-geometry, selection, and factorial summaries.

The original evaluation used the pre-v2 0–4 judge and is retained only as
historical evidence. New comparisons must be rerun with the repository's
versioned Judge v2.

Not committed here:

- 193 MB of direction tensors;
- full model generations and caches;
- 28 MB of raw candidate rows that can be reconstructed from
  `AntonKorznikov/feature_stories`;
- API keys and local environment files.

The scripts are named `legacy_*` because their paths and imports reflect the
original standalone project. The active calm-plus-French experiment lives in
the adjacent `calm-french-composition/` directory and uses the shared
`steering/` and `judge/` packages.
