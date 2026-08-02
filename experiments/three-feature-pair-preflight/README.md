# Three-feature pair preflight

Prepares donor pairs for Russian language, optimism, and atomic sentences
without extracting directions, generating steered answers, or calling Judge.

- Russian uses locally translated, content-matched English sentences.
- Optimism reuses the existing paired corpus.
- Atomic sentences use locally rewritten, content-matched English sentences.

Raw pairs and source data stay in the external output directory. Only a compact
preflight summary may be committed.
