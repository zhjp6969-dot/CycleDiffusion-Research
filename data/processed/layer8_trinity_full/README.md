# Layer 8 Trinity full-grid evidence

This directory contains the integrity-checked, non-audio evidence from the
completed Kaggle run on 2026-09-22.

- 120 ordered speaker-triple/source-sentence units;
- four generated roles per unit;
- 480 generated and 480 evaluated outputs;
- 20 source-speaker/source-sentence clusters;
- epoch-50 base checkpoint and 30 diffusion steps;
- archive SHA-256:
  `58bd273dbc71d414238c76241fcd755dc0a1a40573707cba86e69edeaf0e26db`.

The public repository retains `metrics/` paired summaries, the path-free frozen
manifest, integrity record, and exact analysis scripts. Row-level evaluation,
generation journals/configuration, execution logs, and the downloaded archive
remain available locally but are Git-ignored because they include transcripts,
ASR hypotheses, or private Kaggle paths. Generated audio and private model/data
assets are also excluded.
