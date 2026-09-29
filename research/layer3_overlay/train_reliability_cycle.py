#!/usr/bin/env python3
"""Train corrected and reliability-aware CycleDiffusion decoder variants."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import params
from data_reliability import ReliabilityBatchCollate, ReliabilityVCTKDecDataset
from model.vc import DiffVC
from reliability_cycle import (
    ReliabilityConfig,
    combine_reliability,
    differentiable_convert,
    reliability_components,
    weighted_cycle_l1,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument(
        "--resume",
        help="Optional full-model checkpoint. Omit for a clean decoder initialization.",
    )
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=[
        "uniform", "speaker_only", "content_only", "joint", "hard_gate"
    ], default="joint")
    parser.add_argument("--lambda-cycle", type=float, default=1.0)
    parser.add_argument("--start-epoch", type=int, default=51)
    parser.add_argument("--end-epoch", type=int, default=60)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100,
                        help="Write a rolling resumable state every N optimizer steps; 0 disables it.")
    parser.add_argument("--resume-training-state", type=Path,
                        help="Resume model, optimizer, RNG, epoch, and step from a state file.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cycle-batch-size", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--content-temperature", type=float, default=0.50)
    parser.add_argument("--acoustic-temperature", type=float, default=1.00)
    parser.add_argument("--minimum-weight", type=float, default=0.10)
    parser.add_argument("--hard-gate-threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=params.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0,
                        help="Nonzero value is for smoke tests only.")
    parser.add_argument(
        "--max-total-steps", type=int, default=0,
        help="Stop after exactly this many optimizer steps across epochs; 0 disables it.",
    )
    parser.add_argument("--skip-checkpoint", action="store_true",
                        help="Avoid large model files during smoke validation.")
    return parser.parse_args()


def run_signature(args):
    """Fields that must stay fixed for a controlled resumed run."""
    names = (
        "data_dir", "encoder_checkpoint", "resume", "centroids", "variant",
        "lambda_cycle", "batch_size", "cycle_batch_size", "diffusion_steps",
        "learning_rate", "content_temperature", "acoustic_temperature",
        "minimum_weight", "hard_gate_threshold", "seed", "max_total_steps",
    )
    return {name: str(getattr(args, name)) for name in names}


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["name"], numpy_state["keys"].cpu().numpy(),
        numpy_state["position"], numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_training_state(path, model, optimizer, args, *, epoch, completed_step,
                        global_step, gradient_checked, epoch_generator_state,
                        epoch_complete):
    """Atomically replace one rolling checkpoint, including deterministic resume state."""
    payload = {
        "schema_version": 1,
        "signature": run_signature(args),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "completed_step": completed_step,
        "epoch_complete": epoch_complete,
        "global_step": global_step,
        "gradient_checked": gradient_checked,
        "epoch_generator_state": epoch_generator_state,
        "rng_state": capture_rng_state(),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def trim_metrics_for_resume(path, global_step):
    """Discard log rows newer than the rolling state to prevent duplicates."""
    if not path.exists():
        if global_step:
            raise ValueError("resume state has steps but metrics.jsonl is missing")
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["global_step"]) <= global_step:
            kept.append(json.dumps(record, sort_keys=True))
    if global_step and (not kept or json.loads(kept[-1])["global_step"] != global_step):
        raise ValueError("metrics.jsonl does not contain the saved global step")
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def create_model(device):
    model = DiffVC(
        params.n_mels, params.channels, params.filters, params.heads,
        params.layers, params.kernel, params.dropout, params.window_size,
        params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
        params.beta_min, params.beta_max,
    ).to(device)
    return model


def cycle_gradient_probe(cycle_loss, model):
    probe_parameter = model.decoder.estimator.final_conv.weight
    gradient = torch.autograd.grad(
        cycle_loss, probe_parameter, retain_graph=True, allow_unused=True
    )[0]
    magnitude = 0.0 if gradient is None else float(gradient.abs().sum().item())
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise RuntimeError(
            "cycle gradient probe failed: the reverse-cycle loss is not connected "
            "to decoder.estimator.final_conv.weight"
        )
    return magnitude


def main():
    args = parse_args()
    if args.start_epoch > args.end_epoch:
        raise ValueError("start-epoch must be <= end-epoch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ReliabilityConfig(
        variant=args.variant,
        content_temperature=args.content_temperature,
        acoustic_temperature=args.acoustic_temperature,
        minimum_weight=args.minimum_weight,
        hard_gate_threshold=args.hard_gate_threshold,
    )
    manifest = vars(args).copy()
    manifest["output_dir"] = str(args.output_dir)
    manifest["resume_training_state"] = (
        str(args.resume_training_state) if args.resume_training_state else None
    )
    manifest["reliability"] = asdict(config)
    manifest["cycle_path"] = "first_pass_frozen_second_pass_differentiable"
    dataset = ReliabilityVCTKDecDataset(args.data_dir, args.centroids, seed=args.seed)
    # The dataset uses a local RNG for deterministic ordering; restore the
    # experiment seed for stochastic target/reference selection in __getitem__.
    random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=0, collate_fn=ReliabilityBatchCollate(), generator=generator,
    )
    model = create_model(device)
    encoder_state = torch.load(
        args.encoder_checkpoint, map_location=device, weights_only=True
    )
    model.encoder.load_state_dict(encoder_state, strict=False)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.learning_rate)

    log_path = args.output_dir / "metrics.jsonl"
    gradient_checked = args.lambda_cycle == 0.0
    global_step = 0
    resume_epoch = None
    resume_step = 0
    resume_rng_state = None
    resume_epoch_generator_state = None
    if args.resume_training_state:
        saved = torch.load(
            args.resume_training_state, map_location=device, weights_only=True
        )
        if saved.get("schema_version") != 1:
            raise ValueError("unsupported training-state schema")
        if saved["signature"] != run_signature(args):
            differences = {
                name: {"saved": saved["signature"].get(name), "current": value}
                for name, value in run_signature(args).items()
                if saved["signature"].get(name) != value
            }
            raise ValueError(f"training-state configuration mismatch: {differences}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        global_step = int(saved["global_step"])
        gradient_checked = bool(saved["gradient_checked"])
        resume_rng_state = saved["rng_state"]
        resume_epoch_generator_state = saved["epoch_generator_state"]
        if saved["epoch_complete"]:
            resume_epoch = int(saved["epoch"]) + 1
        else:
            resume_epoch = int(saved["epoch"])
            resume_step = int(saved["completed_step"])
        if resume_epoch < args.start_epoch:
            raise ValueError("training state is older than --start-epoch")
        trim_metrics_for_resume(log_path, global_step)

    # Write the manifest only after a resume state has passed the controlled-field
    # check, so an invalid resume attempt cannot overwrite the accepted config.
    (args.output_dir / "run_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    first_epoch = resume_epoch if resume_epoch is not None else args.start_epoch
    state_path = args.output_dir / "training_state_latest.pt"
    if args.max_total_steps and global_step >= args.max_total_steps:
        if not args.skip_checkpoint:
            torch.save(
                model.state_dict(),
                args.output_dir / f"vc_step{global_step}.pt",
            )
        return
    for epoch in range(first_epoch, args.end_epoch + 1):
        model.train()
        skipped = resume_step if epoch == resume_epoch else 0
        if epoch == resume_epoch and resume_epoch_generator_state is not None:
            generator.set_state(resume_epoch_generator_state.cpu())
        epoch_generator_state = generator.get_state().clone()
        loader_iterator = iter(loader)
        for _ in range(skipped):
            next(loader_iterator)
        if epoch == resume_epoch and resume_rng_state is not None:
            # Replaying the data batches reconstructs the same shuffled order;
            # restoring the saved states then reproduces the next stochastic step.
            restore_rng_state(resume_rng_state)
        remaining = len(loader) - skipped
        if args.max_steps_per_epoch:
            remaining = min(remaining, max(args.max_steps_per_epoch - skipped, 0))
        if args.max_total_steps:
            remaining = min(remaining, max(args.max_total_steps - global_step, 0))
        if remaining <= 0:
            break
        progress = tqdm(
            itertools.islice(loader_iterator, remaining), total=remaining,
            desc=f"epoch {epoch} (from step {skipped + 1})",
        )
        last_step = skipped
        for step, batch in enumerate(progress, start=skipped + 1):
            optimizer.zero_grad(set_to_none=True)
            mel = batch["mel1"].to(device)
            mel_ref = batch["mel2"].to(device)
            lengths = batch["mel_lengths"].to(device)
            source_embedding = batch["c"].to(device)
            target_mel = batch["mel_tgt"].to(device)
            target_lengths = batch["tgt_mel_lengths"].to(device)
            target_embedding = batch["tgt_c"].to(device)
            direct_loss = model.compute_loss(mel, lengths, mel_ref, source_embedding)

            cycle_loss_value = 0.0
            weight_mean = direct_loss.new_ones(())
            part_means = {name: 1.0 for name in ("speaker", "content", "acoustic")}
            gradient_probe = None
            # Backpropagate direct supervision first so its batch-size-four graph
            # does not coexist with all unrolled reverse-cycle graphs on a T4.
            direct_loss.backward()
            if args.lambda_cycle != 0.0:
                count = min(args.cycle_batch_size, mel.shape[0])
                source = mel[:count]
                source_lengths = lengths[:count]
                with torch.no_grad():
                    _, pseudo = model(
                        source, source_lengths, target_mel[:count],
                        target_lengths[:count], target_embedding[:count],
                        n_timesteps=args.diffusion_steps, mode="ml",
                    )
                parts = reliability_components(
                    model, source, source_lengths, pseudo, target_mel[:count],
                    target_lengths[:count], batch["tgt_ref_score"][:count].to(device),
                    config,
                )
                weights = combine_reliability(parts, config)
                weight_mean = weights.mean()
                part_means = {
                    name: float(value.mean().item()) for name, value in parts.items()
                }
                denominator = weights.sum().clamp_min(1e-6)
                # Preserve the batch-weighted objective while materializing only
                # one differentiable reverse sampler at a time. This avoids the
                # 14+ GiB peak observed for three simultaneous six-step graphs.
                for item_index in range(count):
                    _, reconstructed = differentiable_convert(
                        model,
                        pseudo[item_index:item_index + 1],
                        source_lengths[item_index:item_index + 1],
                        source[item_index:item_index + 1],
                        source_lengths[item_index:item_index + 1],
                        source_embedding[item_index:item_index + 1],
                        n_timesteps=args.diffusion_steps,
                        mode="ml",
                    )
                    _, per_example = weighted_cycle_l1(
                        reconstructed,
                        source[item_index:item_index + 1],
                        source_lengths[item_index:item_index + 1],
                        weights[item_index:item_index + 1],
                    )
                    raw_cycle = per_example[0]
                    if not gradient_checked:
                        gradient_probe = cycle_gradient_probe(raw_cycle, model)
                        gradient_checked = True
                    contribution = weights[item_index] * raw_cycle / denominator
                    cycle_loss_value += float(contribution.detach().item())
                    (args.lambda_cycle * contribution).backward()

            total_loss_value = float(direct_loss.item()) + args.lambda_cycle * cycle_loss_value
            grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            record = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "direct_loss": float(direct_loss.item()),
                "cycle_loss": cycle_loss_value,
                "total_loss": total_loss_value,
                "weight_mean": float(weight_mean.item()),
                "speaker_reliability_mean": part_means["speaker"],
                "content_reliability_mean": part_means["content"],
                "acoustic_reliability_mean": part_means["acoustic"],
                "decoder_grad_norm": float(grad_norm.item()),
                "cycle_gradient_probe": gradient_probe,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            progress.set_postfix(total=f"{record['total_loss']:.4f}")
            last_step = step
            if (not args.skip_checkpoint and args.checkpoint_every_steps > 0 and
                    global_step % args.checkpoint_every_steps == 0):
                save_training_state(
                    state_path, model, optimizer, args, epoch=epoch,
                    completed_step=step, global_step=global_step,
                    gradient_checked=gradient_checked,
                    epoch_generator_state=epoch_generator_state,
                    epoch_complete=False,
                )

        if (not args.skip_checkpoint and
                (epoch % args.checkpoint_every == 0 or epoch == args.end_epoch)):
            torch.save(model.state_dict(), args.output_dir / f"vc_{epoch}.pt")
        if not args.skip_checkpoint:
            epoch_complete = last_step == len(loader)
            saved_generator_state = (
                generator.get_state().clone() if epoch_complete
                else epoch_generator_state
            )
            save_training_state(
                state_path, model, optimizer, args, epoch=epoch,
                completed_step=last_step, global_step=global_step,
                gradient_checked=gradient_checked,
                epoch_generator_state=saved_generator_state,
                epoch_complete=epoch_complete,
            )
            if args.max_total_steps and global_step >= args.max_total_steps:
                torch.save(
                    model.state_dict(),
                    args.output_dir / f"vc_step{global_step}.pt",
                )
        resume_step = 0
        resume_rng_state = None
        resume_epoch_generator_state = None
        if args.max_total_steps and global_step >= args.max_total_steps:
            break


if __name__ == "__main__":
    main()
