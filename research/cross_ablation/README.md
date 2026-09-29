# Cross-ablation comparison audit

This directory records the two direct comparisons requested after Layers 4
and 5:

- `content_only_e51 - joint_e51`;
- `content_only_e51 - joint_lambda025_e51`.

The repository contains the Layer 4 and Layer 5 aggregate tables, so their
mean differences are reproducible.  It does **not** contain the 540 Layer 4
component-level evaluation rows or the saved source-cluster differences.  The
Kaggle Version 1 page confirms that the analysis completed, but that version
has zero persisted output files; the former 5.49 GB draft archive was not
retained.  The private asset snapshot contains only the earlier
base/uniform/joint raw table.

Consequently, the output here is descriptive only.  It must not be reported
with a confidence interval or p-value.  A valid paired comparison requires
regenerating the missing `content_only_e51` evaluation rows on the identical
180-item grid (or restoring the original raw table).

Run:

```bash
python research/cross_ablation/analyze_available_means.py
```

The command writes repository-safe outputs to
`data/processed/cross_ablation/`.
