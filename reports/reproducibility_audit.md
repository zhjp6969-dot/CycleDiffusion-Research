# Reproducibility audit

Date: 2026-09-15; latest result update: 2026-09-26

## Audit outcome

The repository reproduces the initial conditioning statistics and figures from
sanitized row-level scores. Later experiments provide safe aggregate tables and
analysis code, but full reanalysis requires excluded raw evaluator records.
The headline verifier checks stored summaries and confirmation unit means;
the overview figures are regenerated directly from the included summaries.
A private, integrity-recorded Colab log and historical
generation snapshot make the 900-output generation protocol auditable,
including the alpha formula, file slices, checkpoint path, diffusion steps, and
seeds. Full end-to-end audio reproduction is still not possible from this
public package because the model weights, VCTK audio, generated waveforms,
embedding arrays, historical upstream source, and a fully pinned environment
are not included.

## Training-gradient audit

The archived training script correctly places the first conversion inside an
explicit `torch.no_grad()` block and appears to leave the second conversion
trainable. Static inspection of the called model code reveals a second,
effective gradient barrier:

| Call path | Recovered decorator |
|---|---|
| `model.vc.DiffVC.forward` | `@torch.no_grad()` |
| `model.diffusion.Diffusion.forward` | `@torch.no_grad()` |
| `model.diffusion.Diffusion.reverse_diffusion` | `@torch.no_grad()` |

Consequently, the reconstructed cycle output and its L1 cycle loss are detached
from the decoder parameters. In the archived implementation, direct diffusion
supervision can update the decoder, but this cycle term cannot. The standalone
auditor in `scripts/audit_cycle_gradient_path.py` reproduces this finding when
pointed to a lawfully obtained local copy of the historical source snapshot.

The continuation overlay in `research/layer3_overlay/` therefore implements a
gradient-enabled second reverse-diffusion path while preserving a frozen first
conversion. It also checks a named decoder parameter for a finite, non-zero
cycle gradient before saving a successful run. GPU checks completed on a Colab
T4 with PyTorch `2.11.0+cu128`: the one-step sampler probe was `1832.31`, and
the preregistered six-step sampler probe was `5979.17` with cycle L1 `1.30681`.
Both runs returned successfully and produced finite decoder gradients. This is
implementation evidence only, not evidence of improved conversion quality.

The original vectorized batch of three six-step cycle graphs exceeded the T4
memory limit (about 14.24 GiB allocated). The accepted implementation first
backpropagates direct supervision, then evaluates each of the three cycle
examples sequentially while retaining the original detached weight denominator.
The three weighted contributions therefore sum to the same batch objective but
only one differentiable sampler is resident at a time. The production-shape
smoke check (batch 4, three cycle examples, six steps) passed in 6.85 seconds
for the training step with cycle-gradient probe `89.22`.

For continuation runs, sentence IDs `002`, `003`, `004`, `005`, `006`, `007`,
`009`, `010`, `011`, and `012` are excluded consistently from source samples,
target-reference sampling, and training centroids. A 50-reference sampling audit
found no excluded ID. Because continuation starts from the archived epoch-50
checkpoint, this prevents new exposure but cannot establish that those utterances
were never used in the checkpoint's earlier training history.

## Verified protocol

| Component | Recovered behavior |
|---|---|
| Speakers | `p236`, `p239`, `p259`, `p263` |
| Directions | All 12 ordered, non-self pairs |
| Source inputs | First 5 sorted WAV files per source speaker |
| Target references | First 3 sorted WAV files per target speaker |
| Main conditioning centroid | First 20 sorted WAV files per target speaker |
| Evaluation centroid | Sorted WAV files 41--60 per speaker |
| Alpha grid | `0`, `0.25`, `0.5`, `0.75`, `1` |
| Interpolation | L2-normalized linear blend of normalized reference and centroid embeddings |
| Diffusion steps | 30 |
| Seed | `1234 + 100 * source_index + reference_index` |
| VC checkpoint path | `real_last_cycle_train_dec_4speakers_original/final_pt/vc_270_0829.pt` |
| Speaker encoder path | `checkpts/spk_encoder/pretrained.pt` |
| Vocoder path | `checkpts/vocoder/` |

The run log reports the speaker encoder as `pretrained.pt` at training step
1,564,501. It records a Tesla T4, CUDA 13.0 in `nvidia-smi`, and PyTorch
`2.11.0+cu128`. Package installs other than PyTorch were unpinned.

## Disjointness finding

The word `disjoint` in the evaluation artifact names refers to the independent
WavLM evaluation centroids. Those centroids use utterances 41--60 and do not
overlap the generation-side utterances.

The main 900-output grid itself is not fully disjoint on the conditioning side:
its centroid uses utterances 1--20, while its target references use utterances
1--3. A later clean-centroid screen uses utterances 21--40 and therefore removes
that overlap, but covers only four directions and two settings (`alpha=0.5` and
`1`). It is not the basis of the main 900-row findings.

This distinction does not invalidate the within-grid sensitivity analysis, but
it narrows the claim: the results diagnose this particular reference-plus-
centroid construction and should not be presented as evidence from a fully
held-out conditioning protocol.

## Reproduction levels

| Level | Status | What is available |
|---|---|---|
| Statistical audit | Complete | Row-key validation, paired anomaly exclusion, clustered LOSO, bootstrap CIs |
| Figures/tables | Complete | Deterministic standard-library analysis from public numeric tables |
| Protocol inspection | Substantially complete | Public hashes and audit records; private notebook and historical source snapshot; seeds, file slices, model paths |
| Audio regeneration | Incomplete | Requires private weights, VCTK audio, generated artifacts, and pinned dependencies |
| Bitwise reproduction | Unsupported | CUDA determinism was not enforced and most packages were unpinned |

## Preserved caveats

The Colab run patched the archived MPS/CPU device selection to use CUDA. The
notebook documents this runtime mutation. The archived inference script also
assigns both variables supplied to its spectral-subtraction routine from the
same generated Mel tensor. That line is preserved rather than silently fixed,
because changing it would produce a new method rather than reconstruct the
completed experiment.

The private notebook copy removes Colab account metadata but retains the
research code, commands, outputs, and timestamps. It is an experiment log, not
a notebook that should be executed top-to-bottom on arbitrary data.

## Formal controlled continuation record

The primary `uniform_e51` and `joint_e51` runs each contain exactly 461 step
records and a 505,154,123-byte final checkpoint. The formal evaluator completed
all 540 expected rows (`3 variants × 12 directions × 5 sources × 3 references`)
without duplicate paired keys. Generation and evaluation used append-only,
fsynced progress journals and skipped validated completed keys on restart.

The paired analysis first averages the 180 paired units within each
source-speaker/source-sentence cluster, then bootstraps the 20 resulting cluster
differences. For `joint - uniform`, WavLM identity delta is `+0.02137` with 95%
CI `[+0.00955, +0.03311]`; robust content error is `-0.00104` with 95% CI
`[-0.02805, +0.02868]`. These are small-sample paired summaries because the 20
clusters share four speakers.

The repository preserves the exact Drive-derived aggregate, comparison, and
training summaries as:

- `data/processed/layer6_formal_eval_variant_summary.csv`;
- `data/processed/layer6_formal_eval_paired_comparisons.csv`;
- `data/processed/layer6_formal_training_summary.csv`;
- `data/processed/layer6_formal_summary.json`.

The private Drive retains checkpoints, waveforms, the 540-row raw evaluation
table, transcripts, and progress journals. They are intentionally not copied
into the public repository.

## Full component-ablation record

Layer 4 added 461-step `speaker_only`, `content_only`, and `hard_gate`
continuations on Kaggle, then evaluated them on the same 180-unit grid. Combined
with the reused base, uniform, and joint rows, the ablation contains 1,080
outputs and no duplicated paired keys. Kaggle notebook Version 1 preserves the
completed outputs; the private archive is 5,486,720,920 bytes. Public summaries
are stored as `data/processed/layer4_full_*`.

The full run exposed two padding edge cases that smoke batches had not reached.
When every source in a batch was shorter than the fixed 128-frame collate width,
`sequence_mask(lengths)` produced a narrower mask, and the differentiable
reconstruction was cropped to the batch maximum rather than the collated source
width. Both paths now use the actual source tensor width. The corrected run
resumed from its rolling training state; completed checkpoints and evaluated
rows were not regenerated.

The component result is positive but specific. Identity gains over uniform were
`+0.03370` for content-only, `+0.02137` for joint, and `+0.01952` for hard gate,
with all three 95% paired cluster intervals above zero. Speaker-only was
`+0.00337` with an interval crossing zero. All component-versus-uniform robust
content-error intervals crossed zero. The current hard-gate implementation still
executes reverse-cycle computation for zero-weight samples, so the ablation is
not evidence of computational savings.

## Independent-cohort confirmation record

Layer 10 used four speakers absent from the discovery cohort (`p225`, `p226`,
`p228`, `p232`) and three independently trained seeds (`17`, `37`, `73`). For
each seed, `uniform_lambda100` and `content_only_lambda100` branched from the same
clean epoch-50 base and completed 461 continuation steps. All ten fixed source,
reference, and evaluation-centroid sentence IDs were excluded from base and
continuation training by construction.

The final evaluator produced all 1,080 expected rows. The analysis averages
direction and reference repeats within 60 seed/source-speaker/source-sentence
units, reports seed-specific and speaker-specific effects, and uses a
seed-then-source-unit hierarchical bootstrap. For `content_only - uniform`, the
pooled identity difference is `+0.01753` with 95% interval
`[-0.00967,+0.03983]`. Seed-specific effects are `+0.03474`, `+0.02902`, and
`-0.01116`. Robust content error is `-0.04038` with interval
`[-0.08307,-0.00340]`. The identity interval crosses zero, so the frozen
promotion criterion is not met.

Kaggle notebook Version 3 preserves the completed run. The downloaded essential
archive passed ZIP integrity checks, contains 1,080 progress records and 1,080
raw evaluator rows, and has SHA-256
`084e9e98391d66e791ec4b345ab7e0e7698163020838fdd8b948c7e51b7b51fb`.
Every bundled file matched the per-file SHA-256 inventory after extraction.
Public-safe aggregate tables are stored under
`data/processed/layer10_confirmation/analysis/`. The raw table, journals,
manifest, and archive remain Git-ignored because they contain private paths,
reference text, or ASR hypotheses.

The local analysis script was rerun from the downloaded 1,080-row raw table in
the bundled scientific Python runtime. All CSV schemas, row order, labels, and
JSON structure matched the Kaggle outputs; the largest numerical difference was
`3.55e-15`, attributable only to floating-point reduction order across runtime
versions. The reported rounded statistics and promotion decision are identical.

## Path-length diagnostic record

Layer 8b used the archived epoch-50 base checkpoint and a frozen 60-unit
one/two/three-hop manifest. Generation completed all 420 evaluated outputs plus
240 internal intermediates; evaluation completed all 420 rows. The primary
analysis collapses to 20 source-speaker/source-sentence clusters. Three-hop
paths exceeded two-hop paths by `+0.10386` in word edit distance (95% cluster
bootstrap CI `[+0.06423,+0.14487]`) and `+0.08510` in signed robust content
error (`[+0.04987,+0.12682]`). Both co-primary intervals exclude zero in the
harmful direction, satisfying the frozen diagnostic rule.

Kaggle notebook Version 1, `Layer 8b complete`, preserves the completed run.
The downloaded essential archive passed ZIP integrity checks, contains 420
generation journal rows and 420 evaluation journal/raw rows, and has SHA-256
`4323bc70791b5b57eea9095445a3b4996285825ed27c4567e35201abb65f0218`.
Every bundled file matched the per-file SHA-256 inventory after extraction.
Public-safe aggregate tables are stored under
`data/processed/layer8_path_length/analysis/`; the archive and extracted raw
records remain Git-ignored because they contain private paths, VCTK text, or ASR
hypotheses. The Kaggle GPU session was stopped after local verification.

This record supports a path-length degradation diagnostic only. Its 20 clusters
share four speakers, the checkpoint may have historical exposure to the fixed
sentences, and no human-listener evidence was collected.
