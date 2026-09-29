# Layers 3–5 training overlay

This overlay is intentionally separate from the historical upstream snapshot,
which is retained privately and is not redistributed here. The archived April
2025 script calls `DiffVC.forward`, `Diffusion.forward`, and
`Diffusion.reverse_diffusion`, all decorated with `torch.no_grad()`. Its reported
cycle loss is therefore disconnected from decoder parameters. With a lawfully
obtained local copy of that snapshot, run
`python scripts/audit_cycle_gradient_path.py --root /path/to/snapshot` to verify
the static call-path finding.

The overlay makes only the second B→A sampler differentiable. The first A→B
pseudo-target generation remains frozen. Reliability is computed per example:

- `speaker`: cosine agreement between the chosen target-reference embedding and
  a centroid built only from training-reference utterances;
- `content`: agreement between frozen internal content encodings of A and A→B;
- `acoustic`: a mel-domain outlier/finite-value check;
- `joint`: geometric mean of the three scores; `hard_gate` thresholds it.

These training signals are not the final evaluators. Final identity/content
reporting remains based on the disjoint WavLM/Whisper protocol documented in the
repository.

Sentence IDs `002`--`012` (excluding `008`) are omitted from continuation source
items, target-reference sampling, and centroid construction. This prevents new
continuation-time leakage into the fixed evaluation subset. The resumed epoch-50
checkpoint predates this correction, so the subset is not claimed to be unseen
through the model's entire training history.

## Colab deployment

Copy the Python overlay files and `launch_controlled_colab.sh` to the extracted
CycleDiffusion root, then build centroids:

```bash
python build_training_centroids.py \
  --embed-dir /content/CycleDiffusion/VCTK_2F2M/embeds \
  --output /content/CycleDiffusion/artifacts/train_centroids.npz
```

Run a one-step gradient smoke test before any experiment:

```bash
python train_reliability_cycle.py \
  --data-dir /content/CycleDiffusion/VCTK_2F2M \
  --encoder-checkpoint /content/CycleDiffusion/checkpts/spk_encoder/enc.pt \
  --resume /content/CycleDiffusion/real_last_cycle_train_dec_4speakers_original/vc_50_0823.pt \
  --centroids /content/CycleDiffusion/artifacts/train_centroids.npz \
  --output-dir /content/CycleDiffusion/runs/smoke_joint \
  --variant joint --lambda-cycle 1 --start-epoch 51 --end-epoch 51 \
  --max-steps-per-epoch 1
```

Success requires a positive `cycle_gradient_probe` in `metrics.jsonl`. The
2026-09-15 Colab T4 check passed with both one and six diffusion steps; the
six-step probe was `5979.17` and cycle L1 was `1.30681`. The corresponding
configs, logs, code snapshot, and centroids are preserved in Drive under
`CycleDiffusion_results/layers3_5/layer3_smoke/`.

The production-shape combination (batch 4, three cycle examples, six reverse
steps) initially exceeded T4 memory when all three autograd graphs were kept at
once. The trainer now backpropagates direct supervision first and then accumulates
the three normalized weighted cycle contributions sequentially. This retains the
same weighted objective while holding one unrolled sampler graph at a time. The
production-shape check passed at about 6.85 seconds per step.

## Completed pre-registered experiment matrix

The current decision boundary and definition of done are in
[`../CONTROLLED_EXPERIMENT.md`](../CONTROLLED_EXPERIMENT.md). It is intentionally
short and task-specific: completed layers are not replayed, and new ideas wait
until the primary controlled comparison is complete.

Keep seed, checkpoint, data split, optimizer, six diffusion steps, batch sizes,
and epoch budget fixed across runs.

1. Layer 3 primary comparison: `uniform` versus `joint`, lambda 1. **Complete.**
2. Layer 4 ablation: `uniform`, `speaker_only`, `content_only`, `joint`, and
   `hard_gate`, lambda 1. **Complete.**
3. Layer 5 necessity: `joint` with lambda 1, 0.5, 0.25, and 0. The zero run is
   direct diffusion supervision only and skips both cycle conversions.
   **Complete.**

Do not choose the best checkpoint on the final test set. Use a fixed epoch or a
separate validation split, then evaluate once with the disjoint protocol.

## Interrupt-safe continuation

Full runs write a rolling `training_state_latest.pt` every 100 optimizer steps
and at every epoch boundary. It contains the model, optimizer, data-order state,
random-number states, epoch, and step. Point the output directory directly at
Drive so a Colab disconnect does not discard it. To continue the same run, repeat
the original command and add:

```bash
--resume-training-state /path/to/run/training_state_latest.pt
```

The trainer rejects changes to controlled fields such as variant, seed, learning
rate, batch sizes, or diffusion steps. `--end-epoch` may be extended without
changing the comparison definition. A final `vc_<epoch>.pt` remains a plain model
checkpoint for evaluation; the training-state file is for continuation only.

`validate_resume_roundtrip.py` performs a real two-step run, resumes it for one
more step, and requires the metrics sequence to be exactly `[1, 2, 3]`. This
round trip passed on the Colab T4 on 2026-09-15. For a Drive-backed controlled
run, the launcher detects an existing rolling state automatically:

```bash
nohup bash launch_controlled_colab.sh uniform 37 > \
  /content/drive/MyDrive/CycleDiffusion_results/layers3_5/full4_controlled/uniform_seed37_e51/console.log \
  2>&1 < /dev/null &
```

The launcher uses a 25-step rolling interval for the Colab screen so a runtime
loss discards only a small amount of GPU work. It refuses unsupported variants,
checks every required asset, exits without overwriting a completed checkpoint,
and resumes from the matching Drive state when present.
