# Layer 9 interaction evidence

This directory is the integrity-checked evidence export for the completed
`uniform/content_only x lambda=1/0.25` interaction experiment.

- `layer9_interaction/metrics/layer9_variant_summary.csv`: four-cell means.
- `layer9_interaction/metrics/layer9_paired_effects.csv`: simple effects and the
  primary difference-in-differences after collapsing to 20 source clusters.
- `layer9_interaction/metrics/layer9_summary.json`: complete machine-readable
  result and inference warning.
- `*/metrics.jsonl`: 461 path-free optimizer-step records for each of the three newly
  trained cells.
- `layer9_interaction_evidence.sha256`: integrity record for the locally retained,
  audio- and checkpoint-free Kaggle evidence bundle. SHA-256:
  `412d452197cfc488e07d4c25c10c81202144138c0767d2678ab1e1c3b02bcbb6`.

The 540 newly evaluated raw rows, progress journals, generation metadata,
run configurations, execution log, and downloaded archive remain available
locally but are Git-ignored because they contain transcript-level fields or
private Kaggle paths. Public aggregate analysis combines them with the 180
reused `uniform_lambda100_e51` records without publishing those raw fields.

The primary identity interaction is `-0.02369` (95% cluster-bootstrap CI
`[-0.03716,-0.01111]`; paired sign-flip `p=0.00202`). The robust-content
interaction is `-0.01068` (CI `[-0.03979,+0.02002]`). At lambda 0.25,
content-only also increases silence and lowers RMS level relative to uniform.
The experiment therefore does not support promoting the content-only/weak-cycle
combination.
