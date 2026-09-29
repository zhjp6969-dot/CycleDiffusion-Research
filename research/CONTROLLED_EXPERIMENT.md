# Completed controlled experiment

**Status: complete (2026-09-15).** Both training runs, the 540-output generation
and evaluation grid, paired analysis, and repository reporting have finished.

## Question

Does reliability-aware `joint` weighting improve over `uniform` cycle training
when every other training and evaluation choice is held fixed?

## Fixed boundary

- four speakers: `p236`, `p239`, `p259`, `p263`;
- same epoch-50 starting checkpoint and clean continuation subset;
- seed, optimizer, batch sizes, six diffusion steps, and epoch budget fixed;
- evaluation sentences excluded from continuation sources, target references, and
  training centroids;
- 12 directions with multiple sources and references;
- WavLM identity, Whisper content, clipping, silence, loudness, and runtime cost;
- no human evaluation and no eight-speaker claim.

The epoch-50 checkpoint may have historical exposure to evaluation utterances.
This experiment therefore tests a controlled continuation, not fully unseen-data
training.

## Completion definition

The comparison is complete only when both `uniform` and `joint` have:

1. finished the same preregistered training budget from the same checkpoint;
2. produced resumable state, final model checkpoint, config, and step metrics;
3. completed the same multi-source, multi-reference 12-direction evaluation;
4. been compared with paired uncertainty intervals and efficiency measurements;
5. been reported whether the result is positive, null, or mixed.

All five conditions are satisfied.

No additional weighting idea or eight-speaker run is started before this boundary
is met. `speaker_only`, `content_only`, and optional `hard_gate` follow only after
the primary comparison is operationally complete.

That follow-up boundary is now also complete. The three component continuations
finished on 2026-09-16 and were evaluated together with the reused base,
uniform, and joint rows. Content-only produced the largest identity gain over
uniform; speaker-only did not. Layer 5 cycle-strength testing is therefore the
next controlled question, and it is now complete. Lambda 0.25 produced the
highest identity mean and a paired source-cluster gain over lambda 1, while
lambda 1 retained the best mean WER/CER. Lambda 0 performed substantially worse
on content, showing that cycle supervision is necessary even though the full
coefficient is not the unique best operating point.

## Layer 5 result

All lambda 0.5, 0.25, and 0 continuations completed the same 461-step budget.
The reused base and lambda-1 records plus the three new models produced 900
evaluated outputs on the same 12-direction, five-source, three-reference grid.

| Contrast | Metric | Mean difference | Cluster-bootstrap 95% CI |
|---|---|---:|---:|
| lambda 0.25 - lambda 1 | identity delta | +0.01690 | [+0.00904,+0.02554] |
| lambda 0.25 - lambda 1 | robust content error | +0.00300 | [-0.02250,+0.03050] |
| lambda 0.5 - lambda 1 | identity delta | +0.00476 | [-0.00569,+0.01515] |
| lambda 0.5 - lambda 1 | robust content error | +0.01221 | [-0.02174,+0.04267] |
| lambda 1 - lambda 0 | robust content error | -0.14914 | [-0.21732,-0.08187] |

This closes the planned four-speaker controlled sequence. Further work should
target independent-speaker replication, clean-from-scratch data isolation, or
efficiency rather than selecting another lambda on the final grid.

## Execution order

1. One complete epoch for `uniform`, then one for `joint`, as a full-dataset
   controlled screen. **Complete: both contain exactly 461 optimizer steps.**
2. Inspect training stability and evaluation outputs without choosing a winner.
   **Training artifacts and final checkpoints validated.**
3. Run the fixed 12-direction, five-source, three-reference evaluation once.
   **Complete: 540/540 outputs generated and evaluated.** The restart-safe
   implementation is in `layer6_formal/`.
4. Report the paired result and update the repository whether it is positive,
   null, or mixed. **Complete: positive identity result with neutral robust
   content difference.**

## Result

The primary `joint - uniform` comparison uses 180 paired generation units,
summarized over 20 source-speaker/source-sentence clusters:

| Metric | Mean difference | Paired cluster-bootstrap 95% CI | Reading |
|---|---:|---:|---|
| WavLM identity delta | +0.02137 | [+0.00955, +0.03311] | joint higher |
| Robust content error | -0.00104 | [-0.02805, +0.02868] | no detected difference |
| Silence fraction | -0.01200 | [-0.01605, -0.00814] | joint less silent |
| RMS dBFS | +1.60216 | [+1.25261, +1.94935] | joint louder |

The identity gain was positive in 15/20 source clusters. Both variants had zero
mean clipping in this grid. Against the epoch-50 base, `joint` improved identity
by `+0.01979` (95% CI `[+0.00389, +0.03538]`) and reduced robust content error
by `-0.20345` (95% CI `[-0.26688, -0.14062]`). `uniform` achieved nearly the
same content reduction but no identity improvement over base.

This is a positive controlled result, not a broad generalization claim. The 20
clusters share four speakers, the epoch-50 starting checkpoint may have prior
sentence exposure, and `joint` took approximately 4058 seconds for 461 steps
versus 534 seconds for `uniform` in the recorded runs.
