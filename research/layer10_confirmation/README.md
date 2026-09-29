# Layer 10: independent-cohort, multi-seed confirmation

Layer 10 is the confirmatory replication stage. It does not introduce another
weighting rule or tune cycle strength on the completed four-speaker grid.

## Frozen question

Does the identity advantage of `content_only, lambda_cycle=1` over the matched
`uniform, lambda_cycle=1` control reproduce across a second four-speaker cohort
and three independently trained seeds without an additional content or acoustic
penalty?

The candidate is `content_only, lambda_cycle=1`; the control is
`uniform, lambda_cycle=1`. The rejected `content_only, lambda_cycle=0.25`
combination is not carried forward.

## What “new speakers” means

This is an **independent-cohort replication**, not zero-shot speaker
generalization. The replication cohort must contain four speakers that do not
overlap the discovery cohort (`p236`, `p239`, `p259`, `p263`). A clean base
model is trained for each seed on the replication cohort, then both continuation
variants branch from that same seed-matched base checkpoint. The evaluation
utterances are excluded from base training, continuation training, conditioning
centroids, and training centroids.

Claiming zero-shot generalization would require leaving the replication speakers
out of model training entirely; that is a different experiment and is not part
of Layer 10.

## Frozen design

- replication cohort: exactly four non-overlapping speakers;
- seeds: `17`, `37`, `73`;
- variants: `uniform_lambda100` and `content_only_lambda100`;
- source sentences: `002`, `004`, `005`, `006`, `007`;
- target references: `003`, `009`, `010`;
- evaluation-centroid sentences: `011`, `012`;
- all ten fixed sentence IDs excluded from every training and training-centroid
  input;
- seed-matched base checkpoint, optimizer, number of steps, diffusion schedule,
  batch construction, and per-item generation seeds;
- 12 directed conversions, five sources, and three references per variant and
  seed.

The final grid contains:

```text
2 variants x 3 seeds x 12 directions x 5 sources x 3 references
= 1,080 evaluated outputs
```

Each seed requires one clean base training followed by two matched 461-step
continuations. A seed is excluded as a whole if either branch fails its gradient
probe, training budget, checkpoint-integrity check, or complete evaluation.

## Primary analysis

The primary contrast is `content_only - uniform` for WavLM target-minus-source
identity delta. Robust content error is co-primary as a non-inferiority guard;
WER, CER, silence fraction, RMS dBFS, clipping, and runtime are mandatory
secondary outcomes.

Repeated directions and references are first collapsed within each
seed/source-speaker/source-sentence unit. The analysis then reports:

1. the paired contrast for each seed separately;
2. the seed-level distribution and all three signs;
3. a hierarchical bootstrap that resamples seeds and then source units;
4. per-speaker effects, without treating 1,080 generated files as independent.

The candidate is promoted only if the identity contrast is positive in at least
two of three seeds, the pooled identity interval excludes zero, the robust
content guard does not show a material penalty, and no systematic silence,
loudness, or clipping regression appears. Otherwise the final report records a
mixed or negative replication.

## Asset gate

Prepare one root with this layout:

```text
DATASET/
  wavs/<speaker>/*_mic1.wav
  mels/<speaker>/*_mel.npy
  embeds/<speaker>/*_embed.npy
  txt/<speaker>/*.txt
```

Then run:

```bash
python research/layer10_confirmation/audit_replication_assets.py \
  --data-dir /path/to/DATASET \
  --output data/processed/layer10_asset_audit.json

python research/layer10_confirmation/build_confirmation_manifest.py \
  --audit data/processed/layer10_asset_audit.json \
  --output data/processed/layer10_confirmation_manifest.csv \
  --summary data/processed/layer10_confirmation_manifest.json
```

If the input is a raw VCTK release rather than the four preprocessed modalities,
first run `preprocess_vctk_cohort.py`. It resamples only the four explicitly
selected non-discovery speakers to mono PCM16 at 22.05 kHz, reproduces the
project's exact 80-bin log-Mel transform, uses the archived speaker encoder
checkpoint `pretrained.pt`, copies transcripts, and writes a restart-safe
SHA-256 progress journal. Example:

`enc.pt` must not be substituted: it is the CycleDiffusion acoustic encoder
state dict and lacks the speaker encoder's `model_state` payload. If
`pretrained.pt` is stored elsewhere, pass it explicitly with
`--encoder-checkpoint /path/to/pretrained.pt`.

```bash
python research/layer10_confirmation/preprocess_vctk_cohort.py \
  --raw-wav-root /kaggle/input/VCTK/WAV_ROOT \
  --raw-txt-root /kaggle/input/VCTK/TXT_ROOT \
  --project-root /kaggle/working/CycleDiffusion \
  --output-dir /kaggle/working/layer10_cohort \
  --speakers SPEAKER_A SPEAKER_B SPEAKER_C SPEAKER_D
```

Candidate raw sources must retain their attribution metadata. The University of
Edinburgh CSTR download page is the authoritative origin; the Kaggle mirror
`mfekadu/english-multispeaker-corpus-for-voice-cloning` describes 109 VCTK
speakers and links the original DOI. A mirror is an input convenience, not a
replacement for recording release/version and license provenance.

The original project archive contains only the discovery speakers, so it was not
reused as an independent cohort. The replacement VCTK 0.80 cohort is now frozen
as `p225`, `p226`, `p228`, and `p232`. Kaggle preprocessing completed 1,365
aligned items; the fail-closed audit passed with per-speaker common/training
counts `231/221`, `356/346`, `366/356`, and `412/402`, respectively. The
deterministic builder produced exactly 1,080 manifest rows. Kaggle Version 1,
`Layer 10 preflight complete`, records the notebook state. The 380.6 MiB cohort
was also uploaded as the private dataset
`jinpeng12353/cyclediffusion-layer-10-preprocessed-cohort`, which is the durable
training input. Formal clean training is now complete for seeds `17`, `37`, and
`73`. Every seed has a clean epoch-50 base and matched 461-step
`uniform_lambda100` and `content_only_lambda100` continuations. Seed 17 is
persisted in `cyclediffusion-layer-10-training-state`; seeds 37 and 73 are
persisted in `cyclediffusion-layer-10-branch-state`. Local essential archives
retain the three base checkpoints, six final branch checkpoints,
configurations, metrics, and SHA-256 manifests. The Kaggle GPU session was
stopped after persistence. The complete 1,080-output evaluation described below
has now finished.

## Clean training launcher

`launch_layer10_kaggle.sh` implements the next stage. For each seed it trains a
50-epoch direct-loss base from the archived acoustic encoder plus a randomly
initialized decoder; it never loads a discovery-cohort VC checkpoint. The
`uniform` and `content_only` branches then start from the same seed-matched base
and each stop at exactly 461 continuation optimizer steps. Rolling training
state makes every run restart-safe. `enc.pt` is correct in this stage because it
is explicitly the frozen DiffVC acoustic encoder; it remains invalid as a
substitute for the separate speaker-embedding `pretrained.pt` used in data
preprocessing.

## Frozen evaluation launcher

`launch_layer10_evaluation_kaggle.sh` runs the remaining confirmation stage:

1. `generate_confirmation_grid.py` generates the manifest-defined 1,080 WAVs
   with paired per-item seeds and restart-safe journaling;
2. `evaluate_confirmation_grid.py` computes disjoint WavLM identity, Whisper
   content, clipping, silence, loudness, duration, and runtime outcomes;
3. `analyze_confirmation.py` first collapses direction/reference repeats to the
   frozen seed/source-speaker/source-sentence unit, then reports per-seed signs,
   per-speaker effects, and the seed-then-source-unit hierarchical bootstrap.

The analysis does not silently invent a numerical non-inferiority margin after
training. It can automatically evaluate the frozen identity rule, but records
the content and acoustic guards for explicit qualitative adjudication because
the preregistered protocol used “material penalty” without a numeric cutoff.

## Kaggle evaluation result (2026-09-26)

The private Kaggle notebook
[`jinpeng12353/cyclediffusion-layer-10-independent-confirmation`](https://www.kaggle.com/code/jinpeng12353/cyclediffusion-layer-10-independent-confirmation)
used all eight required input datasets. The final preflight checked 12 required
assets and completed before generation.

```text
2 variants x 3 seeds x 12 directions x 5 sources x 3 references
= 1,080 generated and evaluated outputs
```

The frozen analysis collapsed the grid to 60
seed/source-speaker/source-sentence units. The primary result was:

| Contrast | Metric | Mean difference | Hierarchical 95% interval | Reading |
|---|---|---:|---:|---|
| content-only - uniform | identity delta | +0.01753 | [-0.00967,+0.03983] | interval crosses zero |
| content-only - uniform | robust content error | -0.04038 | [-0.08307,-0.00340] | lower error |
| content-only - uniform | silence fraction | -0.02064 | [-0.04612,+0.00242] | no systematic penalty |
| content-only - uniform | RMS dBFS | -0.10932 | [-0.94093,+1.15348] | no detected difference |
| content-only - uniform | clip fraction | approximately 0 | [-0.00000021,0] | no clipping regression |

The identity effects by seed were `+0.03474` (seed 17), `+0.02902` (seed 37),
and `-0.01116` (seed 73). Thus the sign criterion passed in two of three seeds,
but the pooled interval did not exclude zero. The frozen identity promotion rule
therefore failed. The robust content proxy improved in every seed and in the
pooled interval, while WER and CER were heterogeneous and their pooled intervals
crossed zero. The correct result label is **mixed replication; candidate not
promoted**, not a confirmed identity improvement.

Kaggle notebook Version 3, `Layer 10 confirmation complete`, preserves the final
state. The essential archive contains the raw 1,080-row evaluation table,
restart journals, generation manifest, six analysis tables/files, and a per-file
SHA-256 inventory. It was downloaded, ZIP-tested, extracted, and hash-verified at:

```text
data/processed/layer10_confirmation/
```

The archive SHA-256 is
`084e9e98391d66e791ec4b345ab7e0e7698163020838fdd8b948c7e51b7b51fb`.
The Kaggle GPU session was stopped after persistence, and the quota monitor was
removed because the run is complete. Audio is intentionally omitted from the
essential local bundle; the stored manifests and evaluator rows retain the full
experimental record needed for analysis.
