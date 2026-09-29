#!/usr/bin/env python3
"""Generate the restart-safe frozen one/two/three-hop path-length grid."""

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


ROLES = (
    "anchor",
    "stochastic",
    "alternate_reference",
    "two_hop_via_a",
    "two_hop_via_b",
    "three_hop_via_a_b",
    "three_hop_via_b_a",
)


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
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    return parser.parse_args()


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


def normalize_sentence_id(value):
    value = str(value).strip()
    return value.zfill(3) if value.isdigit() else value


def valid_wav(path):
    return path.is_file() and path.stat().st_size > 44


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("path-length generation requires CUDA")
    if args.diffusion_steps != 30:
        raise ValueError("the frozen protocol requires exactly 30 diffusion steps")
    checkpoint = args.base_checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    spectral_subtraction, create_model, get_mel, sentence_wav = (
        import_generation_helpers()
    )
    with args.unit_manifest.open(newline="", encoding="utf-8") as handle:
        units = list(csv.DictReader(handle))
    required = {
        "unit_id", "src", "tgt", "mid_a", "mid_b", "source_sentence",
        "intermediate_reference_sentence", "anchor_reference_sentence",
        "alternate_reference_sentence", "anchor_seed", "alternate_seed",
        "first_a_seed", "first_b_seed", "second_a_b_seed", "second_b_a_seed",
        "final_seed",
    }
    if not units or required - set(units[0]):
        raise ValueError(f"invalid manifest columns: {sorted(required - set(units[0]))}")
    if len(units) != 60 or len({row["unit_id"] for row in units}) != 60:
        raise ValueError("the frozen path-length manifest must have 60 unique units")
    speakers = {"p236", "p239", "p259", "p263"}
    for row in units:
        for field in (
            "source_sentence", "intermediate_reference_sentence",
            "anchor_reference_sentence", "alternate_reference_sentence",
        ):
            row[field] = normalize_sentence_id(row[field])
        if {row["src"], row["tgt"], row["mid_a"], row["mid_b"]} != speakers:
            raise ValueError(f"unit must use all four speakers: {row['unit_id']}")

    device = torch.device("cuda")
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
            if key in completed or not valid_wav(path):
                raise ValueError(f"invalid journal entry: {key}")
            completed.add(key)
            completed_rows.append(row)

    config = {
        "schema_version": 1,
        "variant": "base_epoch50",
        "checkpoint": str(checkpoint),
        "unit_manifest": str(args.unit_manifest.resolve()),
        "units": len(units),
        "roles": list(ROLES),
        "diffusion_steps": args.diffusion_steps,
        "sample_rate": 22050,
        "expected_outputs": len(units) * len(ROLES),
        "expected_intermediate_outputs": len(units) * 4,
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
        if not valid_wav(destination):
            raise RuntimeError(f"empty generated output: {destination}")

    model = create_model(device)
    model.load_state_dict(torch.load(
        checkpoint, map_location=device, weights_only=True
    ))
    model.eval()
    total = config["expected_outputs"]
    for unit in units:
        unit_id = unit["unit_id"]
        source = sentence_wav(data_root, unit["src"], unit["source_sentence"])
        first_a = intermediate_dir / f"base_epoch50__{unit_id}__first_a.wav"
        first_b = intermediate_dir / f"base_epoch50__{unit_id}__first_b.wav"
        second_a_b = intermediate_dir / f"base_epoch50__{unit_id}__second_a_b.wav"
        second_b_a = intermediate_dir / f"base_epoch50__{unit_id}__second_b_a.wav"
        if not valid_wav(first_a):
            convert(model, source, unit["mid_a"],
                    unit["intermediate_reference_sentence"],
                    unit["first_a_seed"], first_a)
        if not valid_wav(first_b):
            convert(model, source, unit["mid_b"],
                    unit["intermediate_reference_sentence"],
                    unit["first_b_seed"], first_b)
        if not valid_wav(second_a_b):
            convert(model, first_a, unit["mid_b"],
                    unit["intermediate_reference_sentence"],
                    unit["second_a_b_seed"], second_a_b)
        if not valid_wav(second_b_a):
            convert(model, first_b, unit["mid_a"],
                    unit["intermediate_reference_sentence"],
                    unit["second_b_a_seed"], second_b_a)

        specifications = (
            ("anchor", source, unit["anchor_reference_sentence"], unit["anchor_seed"]),
            ("stochastic", source, unit["anchor_reference_sentence"], unit["alternate_seed"]),
            ("alternate_reference", source, unit["alternate_reference_sentence"], unit["anchor_seed"]),
            ("two_hop_via_a", first_a, unit["anchor_reference_sentence"], unit["final_seed"]),
            ("two_hop_via_b", first_b, unit["anchor_reference_sentence"], unit["final_seed"]),
            ("three_hop_via_a_b", second_a_b, unit["anchor_reference_sentence"], unit["final_seed"]),
            ("three_hop_via_b_a", second_b_a, unit["anchor_reference_sentence"], unit["final_seed"]),
        )
        for role, source_path, reference, seed in specifications:
            key = ("base_epoch50", unit_id, role)
            if key in completed:
                continue
            destination = audio_dir / f"base_epoch50__{unit_id}__{role}.wav"
            convert(model, source_path, unit["tgt"], reference, seed, destination)
            result = {
                **unit,
                "variant": "base_epoch50",
                "role": role,
                "seed": int(seed),
                "path": str(destination),
                "checkpoint": str(checkpoint),
            }
            append_jsonl(journal, result)
            completed.add(key)
            completed_rows.append(result)
            print(f"GENERATED {len(completed_rows)}/{total} {destination.name}", flush=True)

    if len(completed) != total:
        raise RuntimeError(f"incomplete generation: {len(completed)}/{total}")
    intermediates = list(intermediate_dir.glob("*.wav"))
    if len(intermediates) != config["expected_intermediate_outputs"]:
        raise RuntimeError(
            f"incomplete intermediates: {len(intermediates)}/"
            f"{config['expected_intermediate_outputs']}"
        )
    manifest = {"config": config, "rows": completed_rows}
    temporary = output_dir / "generation_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "generation_manifest.json")
    print(f"PATH_LENGTH_GENERATION_COMPLETE {total} {output_dir}")


if __name__ == "__main__":
    main()
