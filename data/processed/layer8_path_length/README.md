# Layer 8b path-length results

Status: complete on 2026-09-26.

The frozen base-checkpoint diagnostic generated and evaluated all 420 planned
outputs across 60 paired units. Analysis collapses repeated directions and path
orders to 20 source-speaker/source-sentence clusters.

The primary three-hop-minus-two-hop contrasts were:

- word edit distance: `+0.10386`, 95% cluster-bootstrap CI
  `[+0.06423,+0.14487]`;
- signed robust content error: `+0.08510`, 95% CI
  `[+0.04987,+0.12682]`.

Both intervals exclude zero in the harmful direction, so the frozen primary
hypothesis is supported on this discovery cohort. This is not a model-promotion,
population-level, or perceptual result: the 20 clusters share four speakers, the
epoch-50 checkpoint may have historical sentence exposure, and no human ratings
were collected.

Public-safe aggregate files are in `analysis/`. The result-only ZIP and extracted
raw records are kept locally but ignored by Git because they contain private
Kaggle paths, VCTK text, or ASR hypotheses. The ZIP passed integrity checks and
has SHA-256:

```text
4323bc70791b5b57eea9095445a3b4996285825ed27c4567e35201abb65f0218
```

Kaggle Version 1, `Layer 8b complete`, preserves the completed notebook. The GPU
session was stopped after local persistence and verification.
