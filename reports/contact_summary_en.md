# Research brief: reproducing and extending CycleDiffusion

**Jinpeng Zheng** · [GitHub](https://github.com/zhjp6969-dot) · September 2026

This independent study follows CycleDiffusion from implementation reconstruction
to controlled training comparisons and new-speaker confirmation. The outcome is
a mixed replication: a reliability-weighted candidate showed an identity benefit
in discovery, but clean three-seed confirmation did not meet the frozen identity
criterion. A separate diagnostic found increasing content divergence with longer
conversion paths.

## Question and implementation

Can intermediate-conversion reliability improve cycle supervision? In the
historical code snapshot available for this study, the second conversion used
inference methods decorated with `torch.no_grad()`. I implemented a differentiable
second reverse-conversion path, kept the first conversion frozen, and verified
finite, non-zero decoder gradients. This observation is specific to that snapshot;
its correspondence with the intended training implementation warrants discussion.

## Evidence and candidate selection

Matched 461-step continuations on four discovery speakers compared uniform,
speaker-only, content-only, joint, and hard-gate weighting. Joint improved WavLM
identity delta over uniform by **+0.02137** (95% interval **[+0.00955,+0.03311]**).
Content-only had the largest discovery gain, **+0.03370** (**[+0.02140,+0.04651]**).
A subsequent weighting-by-cycle-strength interaction experiment did not support
weakening the content-only cycle term. **Content-only at lambda 1** was therefore
frozen for confirmation against uniform at lambda 1.

Confirmation used four non-overlapping speakers, three seeds, seed-matched clean
bases and continuations, and 1,080 evaluated outputs. The identity effect was
**+0.01753**, with an interval crossing zero (**[-0.00967,+0.03983]**). Two seeds
were positive and one negative. The bounded content proxy improved by
**-0.04038** (**[-0.08307,-0.00340]**), while WER and CER intervals crossed zero.
The identity criterion was not met.

![Discovery and confirmation effects](../figures/reliability_effects.png)

Points show paired means and bars show stored 95% bootstrap intervals. Discovery
uses 20 source clusters sharing four speakers; confirmation uses 60 seed/source
units across three seeds and four different speakers. Cohorts and training
histories differ, so this is not a causal between-cohort comparison.

## Conversion-path diagnosis

A frozen 420-output experiment compared direct, two-hop, and three-hop paths on
the discovery cohort. Three-hop minus two-hop increased anchor-relative word
edit distance by **+0.10386** (**[+0.06423,+0.14487]**) and signed robust content
error by **+0.08510** (**[+0.04987,+0.12682]**). Independent-speaker replication
and localization of conversion-leg errors are the next research questions.

## Contribution, limits, and discussion

The contribution combines implementation verification, controlled ablations,
candidate selection, and independent confirmation that preserved the mixed result.
Evaluation uses WavLM, Whisper, and acoustic checks; human listening was not
performed. Discovery checkpoints may have historical evaluation-sentence exposure.
Confirmation excluded fixed evaluation IDs from base and continuation training,
but retained pretrained encoders with their own training histories. Three seeds
and four confirmation speakers limit generalization.

I would welcome feedback on the intended cycle-gradient path and whether
independent replication and conversion-leg error analysis would be useful next
directions.

The [detailed history](research_history_en.md) preserves the experiment sequence.
The [audit](reproducibility_audit.md) documents implementation and data boundaries;
[figure notes](../figures/README.md) identify sources and regeneration commands.
