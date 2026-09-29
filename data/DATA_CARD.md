# Data card

## Dataset scope

The experiment uses four speaker identifiers from VCTK: `p236`, `p239`, `p259`,
and `p263`. The converted grid contains 12 directed speaker pairs, five source
utterances per source speaker, three target-reference conditions per direction,
and five alpha settings, for 900 outputs.

The 900 rows are repeated measurements, not 900 independent observations. The
primary inferential unit is the source-speaker/source-utterance pair, giving 20
source clusters. There are 60 direction/source clusters and 180
direction/source/reference units.

## Public tables

- `public/identity_scores.csv`: WavLM target-minus-source cosine difference.
- `public/content_scores.csv`: word/character edit counts plus WER and CER.
- `public/acoustic_scores.csv`: duration, level, clipping, silence, and DC-offset
  diagnostics.

The public tables remove paths, reference transcripts, and ASR hypotheses. They do
not include audio.

## Formal continuation summaries

`processed/layer6_formal_*` contains repository-safe summaries of a separate
540-output controlled continuation grid: epoch-50 base, epoch-51 `uniform`, and
epoch-51 reliability-aware `joint`, each evaluated on 12 directions, five source
sentences, and three target references. The stored files include variant means,
all paired comparison summaries, training summaries, and a concise JSON result.

The row-level formal table is not public because it contains transcript-level
fields and private artifact references. Its primary paired unit count is 180;
uncertainty is summarized after averaging into 20
source-speaker/source-sentence clusters. These clusters still share only four
speakers and are not 20 independent speakers.

`processed/layer4_full_*` extends the same fixed grid to the complete weighting
ablation. It stores six-variant means, the eight prespecified component-versus-
uniform identity/content comparisons, and a concise provenance JSON. The three
new continuations each contain 461 optimizer steps; base, uniform, and joint are
reused from the earlier formal record rather than retrained.

`processed/layer5_lambda_ablation/` contains the repository-safe summaries for
the fixed joint cycle-strength sweep. It combines the reused base and lambda-1
records with matched lambda 0.5, 0.25, and 0 continuations, for 900 evaluated
outputs. The paired tables average repeated directions and references into the
same 20 source-speaker/source-sentence clusters before calculating uncertainty.
Raw paths, transcripts, hypotheses, generated audio, and checkpoints remain
private and are not copied into this directory.

`processed/layer9_interaction/` retains safe summaries of the completed 2 x 2
`uniform/content_only x lambda=1/0.25` interaction record. It includes 720
evaluation rows (180 per cell) in the local evidence bundle; the Git-tracked
subset contains paired source-cluster effects, four-cell means, and three new
461-step training histories. Generation/evaluation journals and the completion
log remain local. The 594 KiB local evidence archive intentionally excludes
checkpoints and generated audio; its SHA-256 is recorded alongside it. The raw
evaluation CSV retains transcript-level evaluator fields and should not be
republished without checking the applicable dataset and transcript terms.

`processed/layer10_confirmation/analysis/` contains the public-safe aggregate
tables for the completed preregistered confirmation. Its non-overlapping cohort
is `p225/p226/p228/p232`; the ten fixed evaluation IDs were excluded from base
and continuation training. Three clean seeds compare `content_only, lambda=1`
with seed-matched `uniform, lambda=1` across 1,080 outputs, collapsed to 60
seed/source-speaker/source-sentence units. The pooled identity difference was
`+0.01753` with hierarchical 95% interval `[-0.00967,+0.03983]`; the robust
content-error difference was `-0.04038` with interval
`[-0.08307,-0.00340]`. The frozen identity promotion rule was not met.

The full local evidence bundle includes the raw 1,080-row evaluator table,
progress journals, generation manifest, and SHA-256 inventory. It is excluded
from Git because it contains private Kaggle paths, VCTK reference text, and ASR
hypotheses. Generated audio and model checkpoints are not present in the local
result-only bundle.

`processed/layer8_path_length/analysis/` contains the public-safe aggregate
tables for the completed frozen path-length diagnostic. The run generated and
evaluated all 420 planned outputs across 60 paired units and collapsed them to
20 source-speaker/source-sentence clusters. The primary three-hop-minus-two-hop
word-edit contrast was `+0.10386` with 95% cluster-bootstrap interval
`[+0.06423,+0.14487]`; signed robust content error was `+0.08510` with interval
`[+0.04987,+0.12682]`. Both frozen co-primary intervals were harmful.

The complete local result-only archive and extracted records are Git-ignored
because they retain private Kaggle paths, VCTK reference text, or ASR
hypotheses. The public aggregate CSV/JSON files contain numeric summaries only.
The result is a four-speaker discovery-cohort diagnostic; the checkpoint may
have historical sentence exposure, and it is not population-level or perceptual
evidence.

## Metrics

Identity is the converted utterance's cosine similarity to the target centroid
minus its similarity to the source centroid. Higher values indicate a relative
shift toward the target speaker under the selected embedding model; they are not
human similarity ratings.

Content is reported through WER and CER. For the robust multi-objective analysis,
each is bounded to `[0,1]`, then the per-output maximum is used as a conservative
content-error proxy. The uncapped mean WER remains in the processed Pareto table
for traceability. This proxy is an analysis choice, not a standard speech metric.

Acoustic diagnostics are mechanical checks and do not measure naturalness.

## Anomaly policy

A direction/source/reference unit is excluded at every alpha if any of its outputs
has `clip_fraction > 0.01`. This paired rule excludes two units and ten outputs:

- `p259 -> p236`, source 5, reference 2
- `p263 -> p236`, source 5, reference 3

One additional output has minor clipping below the exclusion threshold and remains
in the primary analysis.

## Known limitations

- Only five source utterances are available per conversion direction.
- The formal continuation grid includes only four speakers, and the epoch-50
  starting checkpoint may have historical exposure to its evaluation sentences.
- Layer 9 uses one seed and reuses the existing `uniform, lambda=1` cell rather
  than retraining that cell; its 20 clusters are repeated measurements from only
  four speakers, not a population sample.
- Layer 10 uses three seeds but still contains only four replication speakers.
  Its identity effect was heterogeneous across seeds and the pooled interval
  crossed zero; it does not establish population-level or zero-shot speaker
  generalization.
- Layer 8b uses the original four-speaker discovery cohort. Its path-length
  contrasts are clustered over 20 source units, not 20 independent speakers,
  and the epoch-50 checkpoint may have historical exposure to the fixed
  sentences.
- No human-listener ratings were collected.
- WavLM and Whisper introduce evaluator-specific bias.
- Target-reference sensitivity is descriptive because of the small per-direction
  sample size.
- The recovered main-grid conditioning centroid uses the first 20 sorted target
  utterances and overlaps the first three target-reference utterances. The WavLM
  evaluation centroid uses utterances 41--60 and is held out.
- Generation scripts, seed rules, model paths, and part of the runtime record are
  available, but weights, audio, the complete model tree, and fully pinned
  dependency versions are not public.
- The archived inference script contains an as-run spectral-subtraction quirk
  documented in `reports/reproducibility_audit.md`.
- Before publishing derived audio or transcripts, verify the applicable rights and
  attribution requirements for the exact VCTK release used.

## Intended use

These tables support reproduction of the repository's diagnostic analyses. They
should not be used to claim general voice-conversion quality, unseen-speaker
generalization, or perceptual preference.
