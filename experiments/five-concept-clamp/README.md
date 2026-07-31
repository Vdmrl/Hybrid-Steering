# Experiment #5: five-concept composition and recurrent-state clamp

This experiment tests five deliberately heterogeneous axes:

1. French language;
2. concrete language;
3. optimism;
4. first-person voice;
5. bulleted layout versus paragraph layout.

The primary comparison is matched `rank-1 RSS add` versus `rank-1 RSS clamp`.
The clamp projects each recurrent state onto the five rank-1 feature bases and
softly restores the requested coefficients after every generated token. A
small Gram solve accounts for non-orthogonal feature directions. `rank-4 RSS`
add and clamp are included for the full-five comparison, while the pairwise
extension stays rank 1 to keep the critical path bounded.

## Primary matrix

- baseline;
- five rank-1 raw singleton sanity checks;
- full-five raw add, rank-1/rank-4 RSS add, raw clamp, and rank-1/rank-4
  RSS clamp;
- five leave-one-out compositions for matched RSS add/clamp;
- five flip-one RSS-clamp conditions for selectivity.

The registered extension then fills all ten pairs for the matched rank-1 RSS
add/clamp comparison. It starts only after the primary block is safely complete,
so it can be stopped without weakening the primary result.

Dev uses 32 prompts to select the shared scale and clamp beta from
`{0.2, 0.5, 1.0}` subject to a quality-drop floor. Test uses 128 fixed prompts.
The queue is append-only and resumes by stable task ID.

First-person and bulleted-layout results are exploratory until their rubrics
have human calibration. Bullet/prose donors preserve the same sentences and
differ only in list markup. Raw generations, directions, and Judge artifacts
stay outside Git.
