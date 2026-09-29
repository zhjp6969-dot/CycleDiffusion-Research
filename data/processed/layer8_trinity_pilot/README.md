# Layer 8 Trinity pilot evidence

This directory contains the exact non-audio evidence bundle downloaded from
Kaggle Version 3 on 2026-09-22 for the base epoch-50, source-sentence `002`
Trinity pilot.

- `manifest/`: 24 ordered source/intermediate/target speaker units;
- `metrics/`: paired distances, cluster contrasts, summaries, and JSON result;
- `layer8_trinity_evidence.sha256`: archive integrity record.

The downloaded archive, row-level evaluation, generation records, and execution
logs are retained locally but Git-ignored because they contain transcript-level
fields or transient private Kaggle paths. The archive excludes private VCTK
audio, generated waveforms, and model weights. Re-running
`research/layer8_compositional/analyze_trinity.py` on the local raw evaluation
CSV reproduces all reported values; platform-dependent dot products can differ
in the final floating-point digit (approximately `1e-16`).

This is exploratory evidence with four source clusters, not a confirmatory
population-level result.
