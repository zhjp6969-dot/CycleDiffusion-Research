#!/usr/bin/env python3
"""Generate the frozen Layer 10 independent-cohort confirmation grid.

The CSV manifest is authoritative. Generation is restart-safe: each valid WAV
is journaled immediately, and later invocations skip completed keys.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

import params
from model import DiffVC


KEY_FIELDS = (
    "seed", "variant", "source_speaker", "target_speaker",
    "source_sentence", "reference_sentence",
)
EXPECTED_SEEDS = (17, 37, 73)
EXPECTED_VARIANTS = ("uniform_lambda100", "content_only_lambda100")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--training-state-root", type=Path, required=True)
    parser.add_argument("--branch-state-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(EXPECTED_SEEDS))
    parser.add_argument("--variants", nargs="*", default=list(EXPECTED_VARIANTS))
    return parser.parse_args()


def create_model(device):
    return DiffVC(
        params.n_mels, params.channels, params.filters, params.heads,
        params.layers, params.kernel, params.dropout, params.window_size,
        params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
        params.beta_min, params.beta_max,
    ).to(device)


def get_mel(path, mel_basis):
    wav, _ = librosa.load(path, sr=22050)
    wav = wav[: (wav.shape[0] // 256) * 256]
    wav = np.pad(wav, 384, mode="reflect")
    stft = librosa.stft(
        wav, n_fft=1024, hop_length=256, win_length=1024,
        window="hann", center=False,
    )
    magnitude = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
    return np.log(np.clip(mel_basis @ magnitude, a_min=1e-5, a_max=None))


def noise_median_smoothing(values, width=5):
    result = np.copy(values)
    padded = np.pad(values, width, "edge")
    for index in range(result.shape[0]):
        median = np.median(padded[index:index + 2 * width + 1])
        result[index] = min(padded[index + width + 1], median)
    return result


def archived_spectral_subtraction(mel_synth, spectral_floor=0.02,
                                  silence_window=5, smoothing_window=1):
    mel_len = mel_synth.shape[-1]
    energies = [
        np.sum(np.exp(2.0 * mel_synth[:, index:index + silence_window]))
        for index in range(mel_len - silence_window)
    ]
    minimum = int(np.argmin(energies))
    noise = np.min(
        np.exp(2.0 * mel_synth[:, minimum:minimum + silence_window]), axis=-1
    )
    noise = noise_median_smoothing(noise, smoothing_window)
    denoised = np.copy(mel_synth)
    for index in range(mel_len):
        residual = np.exp(2.0 * mel_synth[:, index]) - noise
        denoised[:, index] = np.log(np.sqrt(np.maximum(
            residual, spectral_floor * noise
        )))
    return denoised


def key(row):
    return tuple(str(row[field]) for field in KEY_FIELDS)


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def wav_path(data_root, speaker, sentence):
    path = data_root / "wavs" / speaker / f"{speaker}_{sentence}_mic1.wav"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_path(args, seed, variant):
    root = args.training_state_root if seed == 17 else args.branch_state_root
    path = root / f"seed{seed}" / variant / "vc_step461.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal Layer 10 generation requires CUDA")

    frame = pd.read_csv(args.manifest_csv, dtype={
        "source_sentence": str, "reference_sentence": str,
    })
    if len(frame) != 1080 or frame.duplicated(list(KEY_FIELDS)).any():
        raise ValueError("Layer 10 manifest must contain 1,080 unique rows")
    if set(frame.seed) != set(EXPECTED_SEEDS):
        raise ValueError(f"unexpected seeds: {sorted(frame.seed.unique())}")
    if set(frame.variant) != set(EXPECTED_VARIANTS):
        raise ValueError(f"unexpected variants: {sorted(frame.variant.unique())}")

    selected_seeds = tuple(args.seeds)
    selected_variants = tuple(args.variants)
    if not set(selected_seeds).issubset(EXPECTED_SEEDS):
        raise ValueError(f"invalid seed filter: {selected_seeds}")
    if not set(selected_variants).issubset(EXPECTED_VARIANTS):
        raise ValueError(f"invalid variant filter: {selected_variants}")
    selected = frame[
        frame.seed.isin(selected_seeds) & frame.variant.isin(selected_variants)
    ].copy()

    root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    journal = output_dir / "generation_progress.jsonl"
    config_path = output_dir / "generation_config.json"
    config = {
        "schema_version": 1,
        "manifest_csv": str(args.manifest_csv.resolve()),
        "expected_outputs": 1080,
        "diffusion_steps": args.diffusion_steps,
        "sample_rate": 22050,
        "postprocess": "archived_self_referenced_spectral_subtraction",
        "paired_generation_seed": "manifest_generation_seed",
        "seeds": list(EXPECTED_SEEDS),
        "variants": list(EXPECTED_VARIANTS),
    }
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing generation config does not match this run")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    completed_rows = []
    completed = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_key = key(row)
            if row_key in completed:
                raise ValueError(f"duplicate generation journal key: {row_key}")
            path = Path(row["path"])
            if not path.exists():
                path = audio_dir / path.name
                row["path"] = str(path)
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate != 22050:
                raise ValueError(f"invalid journaled WAV: {path}")
            completed.add(row_key)
            completed_rows.append(row)

    mel_basis = librosa.filters.mel(
        sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000
    )
    sys.path.insert(0, str(root / "hifi-gan"))
    from env import AttrDict
    from models import Generator as HiFiGAN

    with (root / "checkpts/vocoder/config.json").open() as handle:
        hifigan = HiFiGAN(AttrDict(json.load(handle))).to(device)
    vocoder_state = torch.load(
        root / "checkpts/vocoder/generator", map_location=device, weights_only=True
    )
    hifigan.load_state_dict(vocoder_state["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()

    for (seed, variant), group in selected.groupby(["seed", "variant"], sort=True):
        checkpoint = checkpoint_path(args, int(seed), str(variant))
        model = create_model(device)
        model.load_state_dict(torch.load(
            checkpoint, map_location=device, weights_only=True
        ))
        model.eval()
        for row in group.to_dict("records"):
            row_key = key(row)
            if row_key in completed:
                continue
            src = row["source_speaker"]
            tgt = row["target_speaker"]
            source_sentence = row["source_sentence"]
            reference_sentence = row["reference_sentence"]
            source_mel = torch.from_numpy(get_mel(
                wav_path(data_root, src, source_sentence), mel_basis
            )).float().unsqueeze(0).to(device)
            reference_mel = torch.from_numpy(get_mel(
                wav_path(data_root, tgt, reference_sentence), mel_basis
            )).float().unsqueeze(0).to(device)
            source_lengths = torch.tensor(
                [source_mel.shape[-1]], dtype=torch.long, device=device
            )
            reference_lengths = torch.tensor(
                [reference_mel.shape[-1]], dtype=torch.long, device=device
            )
            embed_path = (
                data_root / "embeds" / tgt /
                f"{tgt}_{reference_sentence}_mic1_embed.npy"
            )
            target_embedding = torch.from_numpy(
                np.load(embed_path)
            ).float().unsqueeze(0).to(device)
            generation_seed = int(row["generation_seed"])
            random.seed(generation_seed)
            np.random.seed(generation_seed)
            torch.manual_seed(generation_seed)
            torch.cuda.manual_seed_all(generation_seed)
            started = time.perf_counter()
            with torch.inference_mode():
                _, converted = model(
                    source_mel, source_lengths, reference_mel,
                    reference_lengths, target_embedding,
                    n_timesteps=args.diffusion_steps, mode="ml",
                )
                mel = archived_spectral_subtraction(
                    converted.squeeze(0).cpu().numpy()
                )
                audio = hifigan(
                    torch.from_numpy(mel).float().unsqueeze(0).to(device)
                ).cpu().squeeze().clamp(-1, 1).numpy()
            elapsed = time.perf_counter() - started
            output = audio_dir / (
                f"seed{seed}__{variant}__{src}_to_{tgt}"
                f"__s{source_sentence}__r{reference_sentence}.wav"
            )
            sf.write(output, audio, 22050)
            result = {
                **{field: row[field] for field in KEY_FIELDS},
                "generation_seed": generation_seed,
                "checkpoint": str(checkpoint),
                "path": str(output),
                "generation_seconds": elapsed,
                "audio_duration_s": len(audio) / 22050,
            }
            append_jsonl(journal, result)
            completed.add(row_key)
            completed_rows.append(result)
            print(
                f"GENERATED {len(completed)}/1080 seed={seed} "
                f"variant={variant} {output.name}", flush=True,
            )
        del model
        torch.cuda.empty_cache()

    all_manifest_keys = {key(row) for row in frame.to_dict("records")}
    if completed == all_manifest_keys:
        ordered = sorted(completed_rows, key=key)
        manifest = {"config": config, "rows": ordered}
        temporary = output_dir / "generation_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output_dir / "generation_manifest.json")
        print(f"LAYER10_GENERATION_COMPLETE 1080 {output_dir}")
    else:
        print(
            f"LAYER10_GENERATION_PARTIAL {len(completed)}/1080 "
            f"selected={len(selected)}", flush=True,
        )


if __name__ == "__main__":
    main()
