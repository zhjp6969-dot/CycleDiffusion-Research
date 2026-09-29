# Layer 8: Trinity compositional-consistency diagnostic

**Status: fixed five-sentence base-checkpoint grid complete on Kaggle: 120
paired units, 480 evaluated outputs, and 20 source clusters.**

Layers 3--5 test pairwise inverse consistency: a conversion followed by its
reverse should recover the source.  This layer asks a different question:
whether a direct path and a two-leg path to the same final speaker produce
consistent outputs.

For distinct speakers `i`, `j`, and `k`, write the stochastic,
reference-conditioned converter as

```text
C_i->k(x; r_k, epsilon)
```

The path quantity compares

```text
direct:   C_i->k(x; r_k, epsilon)
composed: C_j->k(C_i->j(x; r_j, epsilon_1); r_k, epsilon_2)
```

High direct/composed divergence is only evidence of a path-consistency problem
if it exceeds two controls:

1. **stochastic baseline:** direct conversions with the same reference and two
   seeds;
2. **reference baseline:** direct conversions with the same seed and two target
   references.

This distinction prevents ordinary diffusion noise or reference sensitivity
from being mislabeled as a compositional failure.

## Fixed four-speaker protocol

- 24 ordered distinct speaker triples (`4 x 3 x 2`);
- five source sentences per source speaker;
- one intermediate reference, one final anchor reference, and one alternate
  final reference;
- 120 source/triple units and four evaluated roles per unit (480 outputs per
  checkpoint);
- paired seeds stored in the manifest;
- WavLM output-to-output cosine distance and Whisper transcript divergence;
- identity/content changes retained as secondary interpretability metrics;
- analysis collapsed to the same 20 source-speaker/source-sentence clusters
  used in Layers 4--6.

The fixed sentence split is source `002,004,005,006,007`, intermediate
reference `003`, target anchor reference `009`, alternate target reference
`010`, and WavLM evaluation-centroid sentences `011,012`. The evaluation
centroid is therefore disjoint from the source and all conditioning references,
although two centroid utterances per speaker remain a small identity estimate.
The epoch-50 base checkpoint may have historical exposure to these sentences.

The first execution deliberately used only source sentence `002` and the
epoch-50 base checkpoint. That 96-output run was a pipeline and signal check.
The subsequent full execution used all five fixed source sentences and
completed the 120-unit protocol above without changing the model or inference
settings.

Generate the fixed manifest with:

```bash
python research/layer8_compositional/build_trinity_manifest.py \
  --output data/processed/layer8_trinity_manifest.csv
```

On the private audio/weight environment, generate and evaluate the complete
grid with:

```bash
python research/layer8_compositional/generate_trinity_grid.py \
  --project-root /path/to/CycleDiffusion \
  --unit-manifest data/processed/layer8_trinity_manifest.csv \
  --variant-checkpoint base_epoch50=/path/to/vc_50_0823.pt \
  --output-dir /path/to/trinity_full \
  --diffusion-steps 30

python research/layer8_compositional/evaluate_trinity_grid.py \
  --project-root /path/to/CycleDiffusion \
  --generated-dir /path/to/trinity_full \
  --output-dir /path/to/trinity_full_eval
```

After generation and automatic evaluation produce the documented four-role
CSV, run:

```bash
python research/layer8_compositional/analyze_trinity.py \
  --evaluation-csv /path/to/trinity_eval_raw.csv \
  --output-dir /path/to/trinity_metrics
```

The primary quantities are `composed - stochastic` and
`composed - alternate_reference` distances.  Positive intervals indicate that
the two-leg path diverges more than the corresponding direct-path control on
this grid.  They do not establish a population-level theorem or a perceptual
failure.

## Completed full-grid result

The Kaggle run generated and evaluated all 480 planned outputs with 30
diffusion steps. The integrity-checked archive and extracted non-audio evidence
are under
[`data/processed/layer8_trinity_full/`](../../data/processed/layer8_trinity_full/).
The archive SHA-256 is
`58bd273dbc71d414238c76241fcd755dc0a1a40573707cba86e69edeaf0e26db`.

Mean direct-to-control distances were:

| Comparison role | WavLM cosine distance | Word edit distance | Character edit distance | Absolute identity-Delta change | Absolute robust-content change |
|---|---:|---:|---:|---:|---:|
| Alternate target reference | 0.00213 | 0.22140 | 0.15245 | 0.00894 | 0.05009 |
| Stochastic direct conversion | 0.06270 | 0.69800 | 0.49070 | 0.07325 | 0.17918 |
| Two-leg composed conversion | 0.04438 | 0.75711 | 0.53903 | 0.05617 | 0.25589 |

Relative to the alternate-reference baseline, composed conversion was farther
away on all five metrics. The composed-minus-reference difference was
`+0.04225` for WavLM distance (20-cluster bootstrap 95% CI
`[+0.03395,+0.05176]`) and `+0.53571` for word edit distance
(`[+0.47314,+0.59607]`); all 20 source clusters were positive for every metric.

The stochastic control again changes the interpretation. Composed conversion
was less divergent than direct reseeding in WavLM space (`-0.01832`,
`[-0.02810,-0.00728]`) and absolute identity-Delta change (`-0.01707`,
`[-0.02991,-0.00432]`). It was more divergent in word edit distance
(`+0.05911`, `[+0.02893,+0.09277]`), character edit distance (`+0.04833`,
`[+0.02172,+0.07496]`), and robust-content change (`+0.07671`,
`[+0.03234,+0.11821]`). Thus the full grid confirms a content-path effect
beyond ordinary reseeding, while the WavLM and identity-change controls do not
support a metric-independent claim that the composed path is universally less
stable.

The 20 source clusters still share only four speakers, the identity centroid
uses only two utterances per speaker, and the epoch-50 checkpoint may have
historical exposure to the fixed sentences. The result is a paired diagnostic
on this grid, not population-level or perceptual evidence.

## Earlier one-sentence pilot

The Kaggle run generated and evaluated all 96 planned outputs with 30 diffusion
steps. The run archive and its extracted tables are under
[`data/processed/layer8_trinity_pilot/`](../../data/processed/layer8_trinity_pilot/).
The archive SHA-256 is
`6963579f23d863a50e8b80fd05b63fd420ca87ecee6202ec6578a3768b1758d0`.

Mean direct-to-control distances were:

| Comparison role | WavLM cosine distance | Word edit distance | Character edit distance | Absolute identity-Delta change | Absolute robust-content change |
|---|---:|---:|---:|---:|---:|
| Alternate target reference | 0.00290 | 0.13270 | 0.08377 | 0.01058 | 0.03030 |
| Stochastic direct conversion | 0.09356 | 0.68139 | 0.48946 | 0.08263 | 0.18939 |
| Two-leg composed conversion | 0.05786 | 0.76274 | 0.52884 | 0.05739 | 0.26894 |

Relative to the alternate-reference baseline, composed conversion was farther
away on all five metrics. For example, the composed-minus-reference difference
was `+0.05496` for WavLM distance (four-cluster bootstrap interval
`[+0.04128,+0.06864]`) and `+0.63004` for word edit distance
(`[+0.51524,+0.73468]`); all four source clusters were positive.

The stochastic control changes the interpretation. Composed conversion was
*less* divergent than the stochastic direct baseline in WavLM space
(`-0.03570`, interval `[-0.05509,-0.01630]`) while being more divergent in word
space (`+0.08135`, `[+0.01103,+0.15167]`). The character, identity-change, and
robust-content intervals crossed zero. The pilot therefore detects a clear
difference from ordinary reference variation, but it does **not** yet support a
metric-independent claim of compositional path failure beyond diffusion
stochasticity.

Because only four source clusters were present, the pilot sign-flip p-values
had very coarse resolution and were descriptive only. The full grid above
supersedes the pilot for the fixed base-checkpoint claim.

Two compatibility fixes discovered during execution are retained in the local
scripts: VCTK sentence IDs are normalized to three digits, and Whisper uses
direct processor/model inference instead of the incompatible Kaggle
Transformers pipeline path.

## Selection boundary

The existing Layer 4--6 final grid already informed the weighting and lambda
choices, so another Trinity run selected from those results would be
exploratory. The independent-cohort, three-seed Layer 10 confirmation has now
been completed and reported as mixed; the candidate was not promoted.

The next compositional question is therefore path length, not another method
variant. Its frozen one/two/three-hop design, manifest, and restart-safe pipeline
are under [`../layer8_path_length/`](../layer8_path_length/). No path-length
result is claimed until its fixed 420-output evaluation completes.
