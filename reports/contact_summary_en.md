# Research brief: auditing and extending CycleDiffusion for voice conversion

## Motivation

CycleDiffusion addresses the train–inference mismatch of reconstruction-trained
diffusion voice conversion by explicitly learning conversion paths through cycle
consistency. While reproducing and examining this framework, I asked a narrower
diagnostic question: how sensitive are speaker identity and linguistic content to
an additional interpolation between a target-reference condition and a
target-speaker centroid condition?

## Setup

I evaluated four VCTK speakers, all 12 directed conversion pairs, five source
utterances per speaker, three target references per direction, and five
interpolation settings (`alpha = 0, 0.25, 0.5, 0.75, 1`), producing 900 converted
utterances. Speaker identity was evaluated with WavLM speaker embeddings, content
with Whisper WER/CER, and acoustic failures with waveform-level sanity checks.
Inference treats the 20 source utterances as the independent clusters rather than
the 900 repeated outputs.

The recovered generation record makes the intervention explicit: normalized
target-reference and target-centroid embeddings are linearly mixed and then
renormalized. The main conditioning centroid uses the first 20 sorted target
utterances, so it includes the first three target references; the independent
WavLM evaluation centroid instead uses utterances 41--60. I report this overlap
as a design limitation rather than treating the main grid as fully held out.

## Findings

The globally best fixed setting remained `alpha=0`. Direction-specific
leave-one-source-out calibration produced a small mean identity gain of `0.00660`
after paired anomaly exclusion (20-cluster bootstrap 95% CI `[0.00172, 0.01214]`),
but the bounded robust content-error proxy also increased by `0.00856` (95% CI
`[0.00135, 0.01575]`). In addition, all three target references selected the same
descriptive best alpha in only 5 of 12 directions. These results suggest both
direction-dependent and reference-dependent behavior rather than a universal
conditioning strength.

An identity–content Pareto analysis further showed that most directions have
multiple non-dominated settings. Maximizing speaker identity alone is therefore
not sufficient for selecting a conversion condition.

## Interpretation

I treat the interpolation as an inference-time diagnostic, not as an improvement
to the CycleDiffusion training method. The current evidence supports a small
identity–content trade-off and sensitivity to the reference condition. It does not
support a perceptual-quality claim because no human-listener study was completed.

## Controlled reliability-aware continuation

The findings motivate a training question: can conversion-path supervision
account for the reliability of an intermediate conversion without sacrificing
content? During source recovery I found that the archived second conversion also
passes through methods decorated with `torch.no_grad()`, disconnecting its cycle
L1 from decoder parameters. I implemented a corrected path that freezes the first
conversion but differentiates through the second, with explicit gradient probes
and memory-safe accumulation for a T4. The production-shape GPU check passed.

A clean-reference 20-step pilot first produced an inconclusive result. I therefore
kept the question fixed and completed matched 461-step `uniform` and `joint`
continuations followed by a 540-output evaluation: 12 directions, five source
sentences, three references, and the epoch-50 base as a common comparator.

On the final paired grid, `joint - uniform` improved WavLM identity delta by
`0.02137` over 20 source clusters (cluster-bootstrap 95% CI
`[0.00955, 0.03311]`; positive in 15/20 clusters). Robust content error changed
by `-0.00104` (95% CI `[-0.02805, 0.02868]`), providing no evidence of an
additional content penalty. Compared with the epoch-50 base, `joint` also
improved identity by `0.01979` and reduced robust content error by `0.20345`.

I treat this as a positive controlled result rather than a general claim: the 20
source clusters share four speakers, the starting checkpoint may have historical
exposure to evaluation sentences, and the unoptimized reliability computation
made the `joint` epoch about 7.6 times slower than `uniform`.

I then completed the prespecified component ablation by training matched
`speaker_only`, `content_only`, and `hard_gate` continuations and combining them
with the existing base, `uniform`, and `joint` records. `content_only` produced
the largest identity gain over `uniform` (`+0.03370`, 95% CI
`[0.02140, 0.04651]`), followed by `joint` (`+0.02137`) and `hard_gate`
(`+0.01952`); the `speaker_only` interval crossed zero. Robust content-error
intervals crossed zero for every component comparison. This narrows the method
claim: the content-reliability signal appears to drive the identity benefit,
while binary gating offers no demonstrated quality advantage over continuous
joint weighting and has not yet reduced computation in the current code path.

Finally, I completed the prespecified cycle-strength ablation at lambda 1, 0.5,
0.25, and 0. The 900-output analysis was non-monotonic: lambda 0.25 had the
highest identity mean and improved identity over lambda 1 by `0.01690`
(20-cluster bootstrap 95% CI `[0.00904, 0.02554]`), while robust content error
changed by `0.00300` (CI `[-0.02250, 0.03050]`). Lambda 0.5 was statistically
indistinguishable from lambda 1 on identity and content, whereas lambda 0 caused
a large content degradation. Thus cycle supervision is necessary here, but a
full coefficient is not required for the strongest embedding-based identity
result. I treat lambda 0.25 as an identity-oriented operating point and lambda 1
as the conservative content-oriented setting, not as a universally tuned value.

## Interaction and independent confirmation

A frozen `uniform/content_only x lambda=1/0.25` interaction experiment then
showed that weaker cycle supervision did not unlock a content-reliability
advantage. The identity difference-in-differences was `-0.02369` (95% CI
`[-0.03716,-0.01111]`), and the `content_only, lambda=0.25` cell showed more
silence and lower RMS level. I therefore carried only `content_only, lambda=1`
into confirmation against the conservative `uniform, lambda=1` control.

The confirmatory experiment used a non-overlapping VCTK cohort (`p225`, `p226`,
`p228`, `p232`), clean base training for seeds 17, 37, and 73, matched 461-step
continuations, and 1,080 evaluated outputs. After collapsing repeated directions
and references, the analysis contained 60 seed/source-speaker/source-sentence
units. The pooled `content_only - uniform` identity effect was `+0.01753`, but
the hierarchical 95% interval crossed zero (`[-0.00967,+0.03983]`). Effects
were positive for seeds 17 and 37 and negative for seed 73. Robust content error
improved by `-0.04038` (`[-0.08307,-0.00340]`), with no systematic acoustic
regression detected.

The preregistered identity promotion rule was therefore not met. My final claim
is a mixed replication: the discovery result motivated a plausible candidate,
but the clean independent-cohort experiment did not confirm a stable identity
advantage across seeds. This negative boundary is part of the contribution and
prevents presenting a selected four-speaker result as general evidence.

The frozen path-length diagnostic has also completed on the original
four-speaker compositional setup. It changes no weighting rule or cycle
coefficient and retains both intermediate-speaker orders plus direct stochastic
and alternate-reference controls. Across 420 evaluated outputs, the three-hop
minus two-hop effect was `+0.10386` for word edit distance (95% CI
`[+0.06423,+0.14487]`) and `+0.08510` for signed robust content error
(`[+0.04987,+0.12682]`). Both co-primary intervals were harmful, supporting
path-length degradation in this discovery cohort. I treat this as a diagnostic,
not a promoted model or population-level claim, because its 20 clusters share
four speakers and the checkpoint may have historical sentence exposure.

## Current research position

I have therefore frozen the project without selecting a new winner. The strongest
defensible contribution is the complete evidence chain: reproducibility audit,
gradient-path correction, controlled discovery-cohort ablations, a clean
multi-seed replication that preserved a mixed result, and a preregistered
path-length diagnostic. Future work should first replicate the path-length effect
on independent speakers with clean training history; only then should it study
which conversion leg introduces the degradation. I would not run another
post-hoc weighting or lambda search from the current results.
