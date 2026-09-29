# Layer 9: weighting × cycle-strength interaction

This layer tests the smallest experiment needed to separate a content-reliability
effect from a generic cycle-strength effect.  It is a preregistered 2 × 2 design:

| weighting | `lambda_cycle=1.0` | `lambda_cycle=0.25` |
|---|---:|---:|
| uniform | reuse the completed formal-grid evaluation | train and evaluate |
| content only | train and evaluate | train and evaluate |

All newly trained cells use the same epoch-50 checkpoint, clean continuation
data, seed 37, epoch 51, optimizer settings, and evaluation grid.  The existing
`uniform_e51` formal evaluation is renamed `uniform_lambda100_e51`; it is not
regenerated.

The primary estimand is the paired difference-in-differences

```text
(content_only - uniform) at lambda=.25
  - (content_only - uniform) at lambda=1
```

computed after collapsing the 180 paired outputs in each cell to 20 independent
source-utterance clusters.  Percentile cluster-bootstrap intervals and a paired
Monte Carlo sign-flip test are reported.  This experiment is confirmatory only
for this four-speaker continuation setting; clean/new-speaker multi-seed
confirmation remains the next layer.

## Completed result

The Kaggle run completed all three new 461-step continuations, generated and
evaluated 540 new outputs, and combined them with the 180 reused
`uniform_lambda100_e51` records. The resulting table contains 720 outputs and
20 source-speaker/source-sentence clusters.

The primary identity interaction was `-0.02369` (95% cluster-bootstrap CI
`[-0.03716,-0.01111]`; Monte Carlo paired sign-flip `p=0.00202`). The robust
content-error interaction was `-0.01068` with CI `[-0.03979,+0.02002]`.
Weakening cycle supervision therefore removed the identity advantage that
content-only weighting had at lambda 1, rather than amplifying it. At lambda
0.25, content-only minus uniform was `-0.00279` for identity (CI crossing zero),
`+0.05683` for silence fraction, and `-2.696 dBFS` for RMS level. The acoustic
interaction was correspondingly large: `+0.09605` silence fraction and
`-4.506 dBFS`.

This is a negative answer to the proposed synergy hypothesis. It does not erase
the earlier lambda-1 content-only result, but it rules out selecting
`content_only, lambda=0.25` as the final method on this evidence. The next
confirmatory stage should use clean/new speakers and multiple seeds, with
`uniform, lambda=1` as the conservative frozen baseline.

The integrity-checked evidence bundle, raw evaluation table, paired effects,
variant summaries, generation journal, and training logs are stored under
`data/processed/layer9_interaction/`. The evidence archive SHA-256 is
`412d452197cfc488e07d4c25c10c81202144138c0767d2678ab1e1c3b02bcbb6`.

Run on Kaggle with:

```bash
bash /kaggle/working/CycleDiffusion/launch_layer9_kaggle.sh
```
