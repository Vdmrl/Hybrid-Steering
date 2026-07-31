# Experiment #5: five-concept composition and recurrent-state clamp

This experiment replaces the overlapping joy/optimism pair with five more
separable axes:

1. French language;
2. concrete language;
3. principled candor;
4. optimism;
5. first-person voice.

The primary comparison is matched `rank-1 RSS add` versus `rank-1 RSS clamp`.
The clamp projects each recurrent state onto the five rank-1 feature bases and
softly restores the requested coefficients after every generated token. A
small Gram solve accounts for non-orthogonal feature directions. `rank-4 RSS`
is retained only as a bridge to Experiment #4, not as another full factorial.

## Primary matrix

- baseline;
- five rank-1 raw singleton sanity checks;
- full-five raw add, RSS add, rank-4 RSS add, raw clamp, and RSS clamp;
- five leave-one-out compositions for matched RSS add/clamp;
- five flip-one RSS-clamp conditions for selectivity.

The registered extension then fills all ten pairs and all ten triples for the
matched rank-1 RSS add/clamp comparison. It starts only after the primary block
is safely complete, so it can be stopped without weakening the primary result.

Dev uses 32 prompts to select the shared scale and clamp beta from
`{0.2, 0.5, 1.0}` subject to a quality-drop floor. Test uses 128 fixed prompts.
The queue is append-only and resumes by stable task ID.

First-person results are exploratory until its new rubric has human
calibration. Raw generations, directions, and Judge artifacts stay outside
Git.
