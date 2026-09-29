#!/usr/bin/env python3
"""Generate restart-safe direct/control/composed outputs for the Trinity grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def import_generation_helpers():
    try:
        from generate_controlled_grid import (
            archived_spectral_subtraction,
            create_model,
            get_mel,
            sentence_wav,
        )
    except ImportError:
        sibling = Path(__file__).resolve().parents[1] / "layer6_formal"
        sys.path.insert(0, str(sibling))
        from generate_controlled_grid import (
            archived_spectral_subtraction,
            create_model,
            get_mel,
            sentence_wav,
        )
    return archived_spectral_subtraction, create_model, get_mel, sentence_wav


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--unit-manifest", type=Path, required=True)
    parser.add_argument(
        "--variant-checkpoint", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    return parser.parse_args()


def checkpoints(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=PATH, found {item!r}")
        name, raw_path = item.split("=", 1)
        if not name or name in result:
            raise ValueError(f"empty or duplicate variant: {name!r}")
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def output_key(row):
    return row["variant"], row["unit_id"], row["role"]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Trinity generation requires CUDA")
    device = torch.device("cuda")
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    (
        spectral_subtraction,
        create_model,
        get_mel,
        sentence_wav,
    ) = import_generation_helpers()
    checkpoint_map = checkpoints(args.variant_checkpoint)
    with args.unit_manifest.open(newline="", encoding="utf-8") as handle:
        units = list(csv.DictReader(handle))
    required = {
        "unit_id", "src", "mid", "tgt", "source_sentence",
        "intermediate_reference_sentence", "anchor_reference_sentence",
        "alternate_reference_sentence", "anchor_seed", "alternate_seed",
        "first_leg_seed", "second_leg_seed",
    }
    if not units or required - set(units[0]):
        raise ValueError(f"invalid unit manifest columns: {required - set(units[0])}")
    if len({row["unit_id"] for row in units}) != len(units):
        raise ValueError("duplicate Trinity unit IDs")
    for row in units:
        if len({row["src"], row["mid"], row["tgt"]}) != 3:
            raise ValueError(f"unit speakers must be distinct: {row['unit_id']}")

    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    intermediate_dir = output_dir / "intermediate"
    audio_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    journal = output_dir / "generation_progress.jsonl"
    completed_rows, completed = [], set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = output_key(row)
            path = Path(row["path"])
            if key in completed or not path.exists() or path.stat().st_size <= 44:
                raise ValueError(f"invalid journal entry: {key}")
            completed.add(key)
            completed_rows.append(row)

    config = {
        "schema_version": 1,
        "variants": list(checkpoint_map),
        "checkpoints": {key: str(value) for key, value in checkpoint_map.items()},
        "unit_manifest": str(args.unit_manifest.resolve()),
        "units": len(units),
        "roles": ["anchor", "stochastic", "alternate_reference", "composed"],
        "diffusion_steps": args.diffusion_steps,
        "sample_rate": 22050,
        "expected_outputs": len(checkpoint_map) * len(units) * 4,
        "expected_intermediate_outputs": len(checkpoint_map) * len(units),
    }
    config_path = output_dir / "generation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing generation config does not match this run")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    mel_basis = librosa.filters.mel(
        sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000
    )
    sys.path.insert(0, str(root / "hifi-gan"))
    from env import AttrDict
    from models import Generator as HiFiGAN

    with (root / "checkpts/vocoder/config.json").open() as handle:
        hifigan = HiFiGAN(AttrDict(json.load(handle))).to(device)
    state = torch.load(
        root / "checkpts/vocoder/generator", map_location=device, weights_only=True
    )
    hifigan.load_state_dict(state["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    data_root = root / "VCTK_2F2M"

    def convert(model, source_path, target, reference_sentence, seed, destination):
        source_mel = torch.from_numpy(get_mel(source_path, mel_basis)).float()
        source_mel = source_mel.unsqueeze(0).to(device)
        source_lengths = torch.tensor(
            [source_mel.shape[-1]], dtype=torch.long, device=device
        )
        reference_path = sentence_wav(data_root, target, reference_sentence)
        reference_mel = torch.from_numpy(get_mel(reference_path, mel_basis)).float()
        reference_mel = reference_mel.unsqueeze(0).to(device)
        reference_lengths = torch.tensor(
            [reference_mel.shape[-1]], dtype=torch.long, device=device
        )
        embed_path = (
            data_root / "embeds" / target
            / f"{target}_{reference_sentence}_mic1_embed.npy"
        )
        target_embedding = torch.from_numpy(np.load(embed_path)).float()
        target_embedding = target_embedding.unsqueeze(0).to(device)
        set_seed(int(seed))
        with torch.inference_mode():
            _, converted = model(
                source_mel, source_lengths, reference_mel, reference_lengths,
                target_embedding, n_timesteps=args.diffusion_steps, mode="ml",
            )
            mel = spectral_subtraction(converted.squeeze(0).cpu().numpy())
            audio = hifigan(
                torch.from_numpy(mel).float().unsqueeze(0).to(device)
            ).cpu().squeeze().clamp(-1, 1).numpy()
        sf.write(destination, audio, 22050)
        if destination.stat().st_size <= 44:
            raise RuntimeError(f"empty generated output: {destination}")

    total = config["expected_outputs"]
    for variant, checkpoint in checkpoint_map.items():
        model = create_model(device)
        model.load_state_dict(torch.load(
            checkpoint, map_location=device, weights_only=True
        ))
        model.eval()
        for unit in units:
            unit_id = unit["unit_id"]
            source = sentence_wav(
                data_root, unit["src"], unit["source_sentence"]
            )
            intermediate = intermediate_dir / f"{variant}__{unit_id}__first_leg.wav"
            composed_key = (variant, unit_id, "composed")
            if composed_key not in completed and not intermediate.exists():
                convert(
                    model, source, unit["mid"],
                    unit["intermediate_reference_sentence"],
                    unit["first_leg_seed"], intermediate,
                )
            specifications = (
                (
                    "anchor", source, unit["tgt"],
                    unit["anchor_reference_sentence"], unit["anchor_seed"],
                ),
                (
                    "stochastic", source, unit["tgt"],
                    unit["anchor_reference_sentence"], unit["alternate_seed"],
                ),
                (
                    "alternate_reference", source, unit["tgt"],
                    unit["alternate_reference_sentence"], unit["anchor_seed"],
                ),
                (
                    "composed", intermediate, unit["tgt"],
                    unit["anchor_reference_sentence"], unit["second_leg_seed"],
                ),
            )
            for role, source_path, target, reference, seed in specifications:
                key = (variant, unit_id, role)
                if key in completed:
                    continue
                destination = audio_dir / f"{variant}__{unit_id}__{role}.wav"
                convert(model, source_path, target, reference, seed, destination)
                result = {
                    **unit,
                    "variant": variant,
                    "role": role,
                    "seed": int(seed),
                    "path": str(destination),
                    "checkpoint": str(checkpoint),
                }
                append_jsonl(journal, result)
                completed.add(key)
                completed_rows.append(result)
                print(f"GENERATED {len(completed_rows)}/{total} {destination.name}", flush=True)
        del model
        torch.cuda.empty_cache()

    if len(completed) != total:
        raise RuntimeError(f"incomplete generation: {len(completed)}/{total}")
    manifest = {"config": config, "rows": completed_rows}
    temporary = output_dir / "generation_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "generation_manifest.json")
    print(f"TRINITY_GENERATION_COMPLETE {total} {output_dir}")


if __name__ == "__main__":
    main()

