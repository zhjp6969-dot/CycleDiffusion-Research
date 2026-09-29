# Auditing and Extending CycleDiffusion for Voice Conversion

This repository documents a reproducibility audit and a sequence of controlled
extensions to CycleDiffusion voice conversion. The project began by reproducing
the authors' previously public implementation, then moved from implementation
verification to controlled ablation, independent replication, and conversion
path-length diagnosis.

The final result is deliberately not presented as a new universally superior
method. A reliability-aware candidate produced positive evidence in the
four-speaker discovery cohort, but its speaker-identity advantage did not satisfy
the frozen confirmation criterion on a non-overlapping cohort across three
seeds. The repository therefore preserves a mixed replication.

## Research questions

1. Does the cycle-consistency loss in the archived implementation actually
   update the diffusion decoder?
2. Can reliability-aware cycle weighting improve speaker identity without
   increasing linguistic-content error?
3. Does conversion quality degrade as the number of conversion legs increases?

## Main findings

| Question | Result | Boundary |
|---|---|---|
| Gradient path | The archived second conversion also passes through `torch.no_grad()` inference methods, detaching cycle L1 from decoder parameters. A differentiable second reverse-conversion path restored a finite, non-zero decoder gradient. | Applies to the preserved historical snapshot; it is not a claim about every author-side implementation. |
| Discovery-cohort weighting | On the fixed four-speaker grid, `joint - uniform` improved WavLM identity delta by `+0.02137` (cluster-bootstrap 95% CI `[+0.00955,+0.03311]`) with no detected robust-content penalty. | Twenty source clusters share four speakers; the checkpoint may have historical sentence exposure. |
| Independent confirmation | On speakers `p225`, `p226`, `p228`, and `p232` across seeds 17, 37, and 73, `content_only - uniform` identity was `+0.01753`, but the hierarchical 95% interval crossed zero (`[-0.00967,+0.03983]`). Robust content error improved by `-0.04038`. | The preregistered identity promotion rule was not met; the result is a mixed replication. |
| Path length | Three-hop minus two-hop increased word edit distance by `+0.10386` and robust content error by `+0.08510`; both 95% intervals excluded zero in the harmful direction. | Discovery-cohort diagnostic only; no independent-speaker confirmation or human listening test. |

## What is included

- an independently written static gradient-path auditor;
- a memory-safe differentiable second reverse-diffusion training path;
- reliability-component, cycle-strength, and interaction ablations;
- a clean, non-overlapping four-speaker confirmation cohort with three seeds;
- direct, two-hop, and three-hop conversion diagnostics;
- sanitized numeric score tables, aggregate statistics, manifests, and figures;
- English research and reproducibility documentation.

## Start here

- [`reports/contact_summary_en.md`](reports/contact_summary_en.md): concise
  research brief and interpretation of the complete evidence chain.
- [`reports/reproducibility_audit.md`](reports/reproducibility_audit.md):
  implementation, data-boundary, and reproducibility audit.
- [`research/CONTROLLED_EXPERIMENT.md`](research/CONTROLLED_EXPERIMENT.md):
  controlled comparison and definition-of-done record.
- [`docs/provenance.md`](docs/provenance.md): historical source provenance and
  the public/private release boundary.

## Reproduce the public analysis

The primary conditioning audit uses only the Python standard library:

```bash
python scripts/analyze_completed_experiment.py
```

It validates the 900-output grid and regenerates the processed summaries and
two SVG figures. Later experiment directories contain their own analysis entry
points and preregistered protocol records.

```text
data/public/       sanitized row-level numeric scores
data/processed/    aggregate and paired analysis outputs
figures/           deterministic diagnostic figures
reports/           English research and reproducibility reports
research/          controlled protocols and training/evaluation extensions
scripts/           reusable audit and statistical-analysis tools
notebooks/         English-only public execution notebooks
docs/              provenance and release-boundary documentation
```

## Evidence boundaries

- No human-listener evaluation was completed; no perceptual-superiority claim
  is made.
- The public package reproduces statistics and figures from stored scores. It
  does not provide one-command audio regeneration.
- Checkpoints, VCTK-derived audio and transcripts, generated waveforms, raw ASR
  hypotheses, and private runtime paths are excluded.
- The historical upstream source snapshot and original Colab log remain in the
  private research archive because no upstream redistribution license could be
  verified.
- The interpolation experiment is an inference-time diagnostic, not a claimed
  replacement for CycleDiffusion training.
- CycleDiffusion operates on Mel-spectrogram features; this project does not
  describe its cycle objective as waveform-level training.

## Licensing status

This private review repository currently carries no redistribution license.
Original analysis software, documentation, and results will be assigned explicit
licenses only after the remaining upstream-derived boundaries are reviewed.
Absence of a license does not grant permission to reproduce or redistribute the
repository contents.

## Reference

D. Yook, G. Han, H.-P. Chang, and I.-C. Yoo, “CycleDiffusion: Voice Conversion
Using Cycle-Consistent Diffusion Models,” *Applied Sciences*, 14(20), 9595,
2024. <https://doi.org/10.3390/app14209595>
