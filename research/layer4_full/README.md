# Layer 4 full reliability-component ablation

**Status: complete (2026-09-16).** All three new continuations, the combined
1,080-output evaluation, paired analysis, Kaggle output version, and private
artifact archive have finished.

This stage follows the completed primary `joint` versus `uniform` comparison.
It asks which reliability component accounts for the identity gain by completing
the matched `speaker_only`, `content_only`, and `hard_gate` runs.

## Fixed design

- Same epoch-50 starting checkpoint, continuation data, seed 37, optimizer,
  batch sizes, six differentiable reverse steps, and 461 optimizer steps.
- Existing `uniform_e51` and `joint_e51` checkpoints are reused; they are not
  retrained.
- The three new checkpoints use the exact same 12-direction, five-source,
  three-reference evaluation and paired item seeds.
- Evaluation adds 540 new outputs. The combined analysis contains six variants
  and 1,080 rows including the base checkpoint.
- The 20 source-speaker/source-sentence clusters remain the paired analysis
  units and still share only four speakers.

## Restart behavior

`launch_layer4_colab.sh` is safe to re-run after a Colab interruption. Each
training run resumes from its rolling state, completed checkpoints are skipped,
and generation/evaluation append fsynced journals before skipping validated
keys. The completed Layer 6 result directory is read but never modified.

Run from the prepared Colab project root:

```bash
nohup bash /content/CycleDiffusion/launch_layer4_colab.sh \
  > /content/drive/MyDrive/CycleDiffusion_results/layers3_5/full4_controlled/layer4_component_ablation/pipeline.log \
  2>&1 < /dev/null &
```

The order is `speaker_only`, `content_only`, then `hard_gate`. This makes the
cheapest scientifically necessary component comparisons available first while
retaining one restart-safe entry point.

## Result

Relative to `uniform`, mean paired identity delta increased by `+0.00337` for
`speaker_only` (95% CI `[-0.00675,+0.01305]`), `+0.03370` for `content_only`
(`[+0.02140,+0.04651]`), `+0.02137` for `joint`
(`[+0.00964,+0.03308]`), and `+0.01952` for `hard_gate`
(`[+0.00788,+0.03193]`). All four robust-content intervals crossed zero.

The component evidence assigns the clearest contribution to content reliability,
not the target-reference speaker score alone. Hard gating retained an identity
gain but did not improve the aggregate quality means over continuous joint
weighting. It is not an efficiency result: the current trainer still executes
the reverse cycle for zero-weight examples.

The completed private Kaggle artifact is
`/kaggle/working/layer4_component_ablation_outputs.tgz` (5,486,720,920 bytes),
saved with notebook Version 1. Repository-safe summaries live under
`data/processed/layer4_full_*`.

Two short-utterance shape failures were discovered only during the full run.
`reliability_cycle.py` now fixes mask width and reconstruction width to the
collated source tensor width. The run resumed from its rolling checkpoint after
both corrections.
