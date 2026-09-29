# Layer 8b: path-length diagnostic

**Status: complete; 420 generated outputs and 420 evaluation rows verified.**

This follow-up extends the completed Trinity diagnostic without introducing a
new reliability weight or tuning `lambda_cycle`. It asks whether conversion-path
divergence grows when the same source reaches the same target through one, two,
or three conversion legs.

## Frozen question

For the archived four-speaker base checkpoint, does a three-leg path produce
more content divergence from the direct conversion than a two-leg path, after
holding the final target reference and final-leg diffusion seed fixed?

This is a path-length diagnostic, not a model-selection experiment. It retains
the original Layer 8 checkpoint and sentence split so that path length is the
only new experimental factor.

## Fixed design

For every ordered source/target pair, the two remaining speakers are named
`mid_a` and `mid_b` in sorted order. Each source/target/source-sentence unit has
seven evaluated roles:

1. `anchor`: direct one-leg conversion;
2. `stochastic`: direct conversion with a second seed;
3. `alternate_reference`: direct conversion with a second target reference;
4. `two_hop_via_a`: source -> mid_a -> target;
5. `two_hop_via_b`: source -> mid_b -> target;
6. `three_hop_via_a_b`: source -> mid_a -> mid_b -> target;
7. `three_hop_via_b_a`: source -> mid_b -> mid_a -> target.

The final leg of every composed role uses the same target reference and seed as
the direct anchor. Earlier legs use distinct manifest seeds. The two
intermediate orders are both retained so that a path-length effect is not based
on one arbitrarily selected speaker order.

- speakers: `p236`, `p239`, `p259`, `p263`;
- source sentences: `002`, `004`, `005`, `006`, `007`;
- intermediate reference sentence: `003`;
- target anchor reference: `009`;
- alternate target reference: `010`;
- WavLM evaluation-centroid sentences: `011`, `012`;
- checkpoint: archived `base_epoch50` only;
- diffusion steps: 30;
- units: `12 directions x 5 source sentences = 60`;
- evaluated outputs: `60 units x 7 roles = 420`;
- internal, non-evaluated intermediates: `60 units x 4 = 240`.

The epoch-50 checkpoint may have historical exposure to the fixed sentences.
This limitation is inherited from Layer 8 and must remain in every report.

## Completed result

The preregistered primary rule was satisfied on the fixed four-speaker discovery
cohort. After collapsing 60 paired units to 20
source-speaker/source-sentence clusters, the three-hop-minus-two-hop contrasts
were:

- word edit distance: `+0.10386`, 95% cluster-bootstrap CI
  `[+0.06423,+0.14487]`, positive in 18/20 clusters;
- signed robust content error: `+0.08510`, 95% CI
  `[+0.04987,+0.12682]`, positive in 17/20 clusters.

Secondary diagnostics point in the same degradation direction: WavLM cosine
distance increased by `+0.03811` (`[+0.02448,+0.05516]`), silence fraction by
`+0.11207` (`[+0.08387,+0.14052]`), and RMS level changed by `-5.24057 dBFS`
(`[-6.65990,-3.93405]`). Clipping did not show a detected change.

This supports a path-length degradation diagnostic, not a new method or a
population-level claim. The 20 clusters share four speakers, the archived
epoch-50 checkpoint may have historical sentence exposure, and the experiment
contains no human-listener evidence. Kaggle Version 1, `Layer 8b complete`,
preserves the notebook; the GPU session was stopped after download. Public-safe
aggregate tables are under `data/processed/layer8_path_length/analysis/`. The
local result-only archive and extracted raw records remain Git-ignored because
they contain private paths, VCTK text, or ASR hypotheses.

## Primary analysis

Each composed output is compared with the direct anchor. The two two-hop orders
are averaged within a unit, as are the two three-hop orders. The primary
increment is:

```text
mean(three-hop distance from anchor) - mean(two-hop distance from anchor)
```

The two co-primary metrics are:

- anchor-to-output word edit distance;
- signed robust-content-error change relative to the anchor.

WavLM cosine distance, character edit distance, identity-Delta changes, and
acoustic outcomes are secondary. The analysis also compares each path length
with the stochastic and alternate-reference direct controls. Repeated targets
and path orders are collapsed into the same 20
source-speaker/source-sentence clusters used by the original Trinity analysis.

A positive result requires both co-primary three-minus-two intervals to exclude
zero in the harmful direction. Mixed metrics are reported as mixed evidence;
there is no automatic method promotion.

## Build the frozen manifest

```bash
python research/layer8_path_length/build_path_length_manifest.py \
  --output data/processed/layer8_path_length_manifest.csv
```

The expected manifest has exactly 60 unique units. GPU generation, evaluation,
and analysis scripts use append-only progress journals and fail closed on
configuration or cardinality mismatches.

Run the complete private-asset pipeline with:

```bash
bash research/layer8_path_length/launch_path_length_kaggle.sh
```

The launcher validates the checkpoint, vocoder, manifest, and scripts before
starting. It writes restart-safe generation and evaluation journals under
`/kaggle/working/layer8_path_length` by default.

## Boundary after completion

Do not start a new weighting or lambda search from this result. A confirmatory
replication on independent speakers and clean training history would be needed
before generalizing the path-length claim. If replicated, the next scientific
question is mechanism (for example, which leg introduces content loss), not
another post-hoc method variant.
