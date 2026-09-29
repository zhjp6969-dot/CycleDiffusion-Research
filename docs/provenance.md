# Source provenance and release boundary

## Origin

This research began from a code-bearing version of the authors' CycleDiffusion
GitHub repository. The repository was cloned locally and used for the first
model-construction, inference, and gradient-path checks. The working tree,
weights, and four-speaker assets were later moved into Colab and preserved in a
Google Drive archive named `CycleDiffusion.zip`.

The current upstream public tree no longer exposes that historical source.
Because the exact removed commit or branch and a redistribution license cannot
be verified, this repository does not republish the upstream source snapshot.
The private copy is retained as audit evidence rather than treated as newly
authored project code.

## Public evidence

The public repository includes:

- independently written gradient-audit and statistical-analysis tools;
- the differentiable second reverse-conversion training extension;
- controlled experiment protocols and launchers;
- sanitized numeric score tables and aggregate results;
- figures, manifests, integrity hashes, and English documentation.

`data/processed/reproduction_manifest.json` records the private notebook and
historical generation-file hashes used during reconstruction. These entries are
provenance records, not links to redistributed source files.

## Private evidence

The following remain outside Git:

- the historical upstream source and generation scripts;
- the original Colab experiment log and non-English research notes;
- model checkpoints, speaker-encoder and vocoder weights;
- VCTK audio, Mel features, embeddings, transcripts, and generated audio;
- raw evaluator output containing local paths, transcripts, or ASR hypotheses.

## Interpretation

The implementation finding is limited to the preserved historical snapshot:
the second conversion passes through inference methods decorated with
`torch.no_grad()`, detaching the reported cycle loss from decoder parameters.
This repository does not claim that every private author version, later version,
or description in the paper shares the same behavior.
