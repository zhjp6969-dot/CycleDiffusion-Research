# Layer 5 full cycle-strength ablation

**Status: complete (2026-09-22).** This layer asks whether the observed
identity result needs full cycle supervision or survives with a weaker cycle
coefficient.

## Fixed design

- `joint` reliability weighting at `lambda_cycle = 1, 0.5, 0.25, 0`;
- lambda 1 reuses the completed `joint_e51` formal record;
- lambda 0.5, 0.25, and 0 start from the same epoch-50 checkpoint and each run
  exactly 461 optimizer steps with seed 37;
- lambda 0 skips both cycle conversions and is the direct-supervision control;
- all variants use the same 12 directions, five source sentences, three target
  references, generation seeds, WavLM/Whisper metrics, and 20 paired source
  clusters.

The primary necessity contrast is lambda 1 versus lambda 0. Intermediate
coefficients test whether the identity/content response is monotonic or whether
the full coefficient is stronger than needed. No checkpoint is selected on the
final grid.

## Run

On Kaggle, after installing the overlay and exporting `CYCLEDIFFUSION_ROOT` and
`CYCLEDIFFUSION_RESULTS_ROOT`:

```bash
bash /kaggle/working/CycleDiffusion/launch_layer5_kaggle.sh
```

Training, generation, and evaluation are restart-safe. The final archive omits
the redundant rolling optimizer states but retains final checkpoints, configs,
step metrics, generated audio, raw evaluation rows, and paired summaries.

## Result

All three new continuations completed 461 optimizer steps. Together with the
reused epoch-50 base and lambda-1 joint model, the fixed grid contains 900
evaluated outputs. Means are:

| Variant | Identity delta | Robust content error | WER | CER |
|---|---:|---:|---:|---:|
| base epoch-50 | 0.29893 | 0.55907 | 0.56197 | 0.37049 |
| lambda 0 | 0.30492 | 0.50477 | 0.57310 | 0.39359 |
| lambda 0.25 | **0.33562** | 0.35863 | 0.49700 | 0.33641 |
| lambda 0.5 | 0.32347 | 0.36784 | 0.36723 | 0.22643 |
| lambda 1 | 0.31871 | **0.35563** | **0.35544** | **0.22096** |

The paired analysis first averages the 180 repeated generation units into 20
source-speaker/source-sentence clusters. Relative to lambda 0, lambda 0.5
improves identity by `+0.01855` (95% bootstrap CI `[+0.00263,+0.03374]`)
and lambda 0.25 by `+0.03070` (`[+0.01406,+0.04681]`); both also reduce robust
content error. Lambda 1 has a clear content advantage over lambda 0, but its
identity interval crosses zero.

Relative to lambda 1, lambda 0.25 improves identity by `+0.01690`
(`[+0.00904,+0.02554]`) while robust content error changes by only `+0.00300`
(`[-0.02250,+0.03050]`). Its mean WER and CER are higher, but their cluster
intervals also cross zero because a few source clusters are highly variable.
Lambda 0.5 and lambda 1 have no detected identity or content difference.

The result is therefore non-monotonic: cycle supervision is necessary for the
large content improvement, but the full coefficient is not necessary for the
highest embedding-based identity score on this grid. Lambda 0.25 is the best
identity operating point, whereas lambda 1 remains the conservative content
operating point. This is a single-seed, four-speaker result with 20 source
clusters and possible historical evaluation-sentence exposure in the epoch-50
checkpoint; it is not a population-level hyperparameter recommendation.

Repository-safe aggregate, cluster, training, and JSON summaries are stored in
`data/processed/layer5_lambda_ablation/`. The private raw table, transcripts,
audio, and checkpoints are excluded from the public repository.
