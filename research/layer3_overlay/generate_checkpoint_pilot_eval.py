#!/usr/bin/env python3
"""Generate a fixed held-out-sentence pilot grid for checkpoint comparison.

This script is designed for the private Colab project tree. It evaluates all 12
directions with one fixed source sentence and one fixed target reference for the
epoch-50 base, uniform-cycle pilot, and joint-reliability pilot checkpoints.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import params
from model import DiffVC


SPEAKERS = ["p236", "p239", "p259", "p263"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--uniform-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sentence", default="002")
    parser.add_argument("--reference-sentence", default="003")
    parser.add_argument("--diffusion-steps", type=int, default=30)
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
    """Preserve the archived self-referenced postprocessing for comparability."""
    mel_source = mel_synth
    mel_len = mel_source.shape[-1]
    energies = [
        np.sum(np.exp(2.0 * mel_source[:, index:index + silence_window]))
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
        signal = np.maximum(residual, spectral_floor * noise)
        denoised[:, index] = np.log(np.sqrt(signal))
    return denoised


def sentence_wav(data_root, speaker, sentence):
    matches = sorted((data_root / "wavs" / speaker).glob(
        f"{speaker}_{sentence}_mic1.wav"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"expected one WAV for {speaker}/{sentence}: {matches}")
    return matches[0]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("pilot generation requires CUDA")
    root = args.project_root.resolve()
    data_root = root / "VCTK_2F2M"
    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    checkpoints = {
        "base_epoch50": args.base_checkpoint,
        "uniform_pilot20": args.uniform_checkpoint,
        "joint_pilot20": args.joint_checkpoint,
    }
    rows = []
    for variant, checkpoint in checkpoints.items():
        model = create_model(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        for src_index, src in enumerate(SPEAKERS):
            source_path = sentence_wav(data_root, src, args.source_sentence)
            source_mel = torch.from_numpy(get_mel(source_path, mel_basis)).float()
            source_mel = source_mel.unsqueeze(0).to(device)
            source_lengths = torch.tensor(
                [source_mel.shape[-1]], dtype=torch.long, device=device
            )
            for tgt_index, tgt in enumerate(SPEAKERS):
                if src == tgt:
                    continue
                reference_path = sentence_wav(
                    data_root, tgt, args.reference_sentence
                )
                reference_mel = torch.from_numpy(
                    get_mel(reference_path, mel_basis)
                ).float().unsqueeze(0).to(device)
                reference_lengths = torch.tensor(
                    [reference_mel.shape[-1]], dtype=torch.long, device=device
                )
                embed_path = (
                    data_root / "embeds" / tgt /
                    f"{tgt}_{args.reference_sentence}_mic1_embed.npy"
                )
                target_embedding = torch.from_numpy(np.load(embed_path)).float()
                target_embedding = target_embedding.unsqueeze(0).to(device)
                seed = 20260915 + 100 * src_index + tgt_index
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    _, converted = model(
                        source_mel, source_lengths, reference_mel,
                        reference_lengths, target_embedding,
                        n_timesteps=args.diffusion_steps, mode="ml",
                    )
                    converted_np = converted.squeeze(0).cpu().numpy()
                    postprocessed = archived_spectral_subtraction(converted_np)
                    audio = hifigan(
                        torch.from_numpy(postprocessed).float().unsqueeze(0).to(device)
                    ).cpu().squeeze().clamp(-1, 1).numpy()
                output = args.output_dir / f"{variant}__{src}_to_{tgt}.wav"
                sf.write(output, audio, 22050)
                rows.append({
                    "variant": variant,
                    "src": src,
                    "tgt": tgt,
                    "source_sentence": args.source_sentence,
                    "reference_sentence": args.reference_sentence,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "path": str(output),
                })
        del model
        torch.cuda.empty_cache()
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps({
            "scope": "descriptive_12_direction_pilot",
            "postprocess": "archived_self_referenced_spectral_subtraction",
            "rows": rows,
        }, indent=2, sort_keys=True) + "\n"
    )
    print(f"generated {len(rows)} files in {args.output_dir}")


if __name__ == "__main__":
    main()
