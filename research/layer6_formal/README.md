# Layer 6 formal controlled evaluation

This directory closes the fixed four-speaker `uniform` versus `joint` question.
It does not add a new method, human evaluation, or an eight-speaker claim.

**Run status: complete.** Generation, evaluation, and analysis finished for all
540 outputs on 2026-09-15.

## Fixed grid

- variants: epoch-50 base, epoch-51 uniform, epoch-51 joint;
- 12 directed speaker conversions;
- five source sentences: `002`, `004`, `005`, `006`, `007`;
- three target references: `003`, `009`, `010`;
- two separate WavLM centroid sentences: `011`, `012`;
- 30 diffusion steps and paired per-item seeds;
- 540 generated and evaluated outputs in total.

All ten sentence IDs were excluded from the continuation loader. The evaluation
centroid does not reuse a source or conditioning reference. The historical
epoch-50 checkpoint caveat still applies, so this remains a controlled
continuation experiment rather than a fully unseen-data experiment.

## Restart behavior

Generation and evaluation each append an fsynced JSONL journal after every
completed item. Re-running the launcher validates and skips completed keys.
Neither phase deletes existing WAVs or accepted metrics.

Run on the prepared Colab runtime:

```bash
nohup bash /content/CycleDiffusion/launch_formal_eval_colab.sh \
  > /content/drive/MyDrive/CycleDiffusion_results/layers3_5/full4_controlled/formal_evaluation/pipeline.log \
  2>&1 < /dev/null &
```

The final analysis uses the 20 speaker/source-sentence units as paired bootstrap
clusters. Because these units still share four speakers, uncertainty intervals
must be described as small-sample paired summaries rather than population-level
confirmatory inference.

## Final result

`joint - uniform` increased WavLM identity delta by `+0.02137` across the 20
paired source clusters (bootstrap 95% CI `[+0.00955, +0.03311]`) while the
robust content-error difference was `-0.00104` (95% CI
`[-0.02805, +0.02868]`). The result is therefore positive for identity and null
for additional robust content cost on this grid. `joint` was also 1.60 dB louder
and had 0.012 less silence fraction on average, so the identity interpretation
must remain accompanied by the acoustic summary.

Repository copies of the aggregate, paired, and training summaries are stored
under `data/processed/` with the `layer6_formal_` prefix. Raw audio, private
checkpoints, and the 540-row transcript-level table remain in Drive.
