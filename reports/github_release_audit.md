# GitHub release audit

Audit date: 2026-09-26

## Outcome

The repository is technically consistent and the public/private evidence
boundary is enforced. It is ready for a final human choice of license and
repository destination; no remote push or public release has been performed.

## Automated checks

- parsed 62 Python files with the Python AST;
- parsed 37 public JSON files;
- opened and validated headers for 50 public CSV files;
- passed `bash -n` for every shell launcher under `research/` and `scripts/`;
- found no broken relative Markdown links in public documentation;
- passed `git diff --check`;
- cross-checked the Layer 8b primary values against
  `path_length_summary.json`;
- verified the Layer 8b result archive SHA-256 and every internal inventory
  hash before keeping the archive outside Git.

## Public/private boundary

The public tree retains protocols, source code, path-free manifests, aggregate
statistics, figures, data-card limitations, and integrity hashes. The following
remain local and are excluded through `.gitignore`:

- model checkpoints, audio, and Kaggle upload payloads;
- row-level evaluation tables containing VCTK reference text or ASR hypotheses;
- generation/evaluation progress journals and run configurations containing
  private runtime paths;
- downloaded evidence archives and execution logs.

Files removed from the Git index during this audit were not deleted from disk.
The historical upstream source, private Colab log, and archived generation
scripts remain in the private workspace. They are represented publicly only by
provenance notes and integrity metadata because no upstream redistribution
license could be verified.

## Scientific release boundary

- Layer 10 is a mixed independent-cohort replication; the candidate method is
  not promoted because the pooled identity interval crosses zero.
- Layer 8b supports path-length degradation in the fixed four-speaker discovery
  cohort, not a population-level or perceptual claim.
- No human-listener evaluation is reported.
- VCTK-derived audio and transcript redistribution remains subject to the exact
  dataset release terms and is not part of this repository.

## Remaining owner decisions

1. Choose and add a software/data license. No license was inferred automatically.
2. Confirm the final GitHub repository name and visibility.
3. Optionally add author metadata (`CITATION.cff`, ORCID, affiliation) after the
   preferred public identity is confirmed.
4. Review the staged diff, then commit and push explicitly.
