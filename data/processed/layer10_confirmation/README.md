# Layer 10 essential result bundle

This directory contains the locally persisted result-only archive from the
completed independent-cohort, three-seed Layer 10 confirmation.

- archive: `layer10_essential_results.zip`
- archive SHA-256:
  `084e9e98391d66e791ec4b345ab7e0e7698163020838fdd8b948c7e51b7b51fb`
- evaluation rows: 1,080
- analysis units: 60 seed/source-speaker/source-sentence units
- result: mixed replication; `content_only_lambda100` is not promoted

The public-safe aggregate tables are mirrored under `analysis/`. The full
extracted files remain locally under `results/layer10_confirmation/`. The
per-file sizes and SHA-256 digests are recorded in
`results/layer10_confirmation/essential_results_inventory.json` and were
verified after download. The archive and `results/` tree are ignored by Git
because the raw evaluator table contains private Kaggle paths, VCTK reference
text, and ASR hypotheses. The archive intentionally excludes generated audio;
it retains the raw evaluator table, progress journals, generation manifest, and
all analysis outputs on this computer.

Primary pooled contrast (`content_only - uniform`):

| Metric | Mean difference | Hierarchical 95% interval |
|---|---:|---:|
| identity delta | +0.01753 | [-0.00967,+0.03983] |
| robust content error | -0.04038 | [-0.08307,-0.00340] |

The identity effect was positive in two of three seeds, but the pooled interval
crossed zero, so the frozen identity promotion criterion was not met.

Reanalysis from the raw 1,080-row table reproduced the stored tables and JSON
semantically. The maximum cross-runtime floating-point difference was
`3.55e-15`; all reported rounded values and the promotion decision matched.
