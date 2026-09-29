#!/usr/bin/env python3
"""Generate the preregistered full four-speaker checkpoint comparison grid.

The script is restart-safe: every completed WAV is recorded immediately in a
JSONL journal. Re-running the same command skips validated completed keys and
continues from the first missing item.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import params
from model import DiffVC


SPEAKERS = ("p236", "p239", "p259", "p263")
SOURCE_SENTENCES = ("002", "004", "005", "006", "007")
REFERENCE_SENTENCES = ("003", "009", "010")
EVAL_CENTROID_SENTENCES = ("011", "012")
DEFAULT_VARIANTS = ("base_epoch50", "uniform_e51", "joint_e51")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--uniform-checkpoint", type=Path)
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument(
        "--variant-checkpoint", action="append", default=[], metavar="NAME=PATH",
        help=("Repeat for an arbitrary matched grid. This is mutually exclusive "
              "with the three legacy checkpoint arguments."),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def checkpoint_map(args):
    legacy = (args.base_checkpoint, args.uniform_checkpoint, args.joint_checkpoint)
    if args.variant_checkpoint:
        if any(value is not None for value in legacy):
            raise ValueError(
                "--variant-checkpoint cannot be mixed with legacy checkpoint arguments"
            )
        checkpoints = {}
        for item in args.variant_checkpoint:
            if "=" not in item:
                raise ValueError(f"expected NAME=PATH, found: {item}")
            name, raw_path = item.split("=", 1)
            if not name or name in checkpoints:
                raise ValueError(f"empty or duplicate variant name: {name!r}")
            checkpoints[name] = Path(raw_path).resolve()
        if not checkpoints:
            raise ValueError("at least one --variant-checkpoint is required")
        return checkpoints
    if any(value is None for value in legacy):
        raise ValueError(
            "provide all three legacy checkpoints or repeated --variant-checkpoint"
        )
    return {
        name: path.resolve()
        for name, path in zip(DEFAULT_VARIANTS, legacy)
    }


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


def record_key(row):
    return (
        row["variant"], row["src"], row["tgt"],
        row["source_sentence"], row["reference_sentence"],
    )


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal generation requires CUDA")
    root = args.project_root.resolve()
    data_root = root / "VCTK_2F2M"
    output_dir = args.output_dir.resolve()
    wav_dir = output_dir / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    journal = output_dir / "generation_progress.jsonl"
    config_path = output_dir / "generation_config.json"
    checkpoints = checkpoint_map(args)
    config = {
        "schema_version": 1,
        "speakers": list(SPEAKERS),
        "source_sentences": list(SOURCE_SENTENCES),
        "reference_sentences": list(REFERENCE_SENTENCES),
        "eval_centroid_sentences": list(EVAL_CENTROID_SENTENCES),
        "variants": list(checkpoints),
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
        "sample_rate": 22050,
        "postprocess": "archived_self_referenced_spectral_subtraction",
        "expected_outputs": (
            len(checkpoints) * len(SPEAKERS) * (len(SPEAKERS) - 1)
            * len(SOURCE_SENTENCES) * len(REFERENCE_SENTENCES)
        ),
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError("existing generation config does not match this run")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    completed_rows = []
    completed = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = record_key(row)
            if key in completed:
                raise ValueError(f"duplicate generation journal key: {key}")
            path = Path(row["path"])
            if not path.exists() or path.stat().st_size <= 44:
                raise ValueError(f"journaled WAV is missing or empty: {path}")
            completed.add(key)
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

    total = len(checkpoints) * len(SPEAKERS) * (len(SPEAKERS) - 1)
    total *= len(SOURCE_SENTENCES) * len(REFERENCE_SENTENCES)
    for variant, checkpoint in checkpoints.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = create_model(device)
        model.load_state_dict(torch.load(
            checkpoint, map_location=device, weights_only=True
        ))
        model.eval()
        for src_index, src in enumerate(SPEAKERS):
            for tgt_index, tgt in enumerate(SPEAKERS):
                if src == tgt:
                    continue
                for source_index, source_sentence in enumerate(SOURCE_SENTENCES):
                    source_path = sentence_wav(data_root, src, source_sentence)
                    source_mel = torch.from_numpy(
                        get_mel(source_path, mel_basis)
                    ).float().unsqueeze(0).to(device)
                    source_lengths = torch.tensor(
                        [source_mel.shape[-1]], dtype=torch.long, device=device
                    )
                    for reference_index, reference_sentence in enumerate(REFERENCE_SENTENCES):
                        key = (variant, src, tgt, source_sentence, reference_sentence)
                        if key in completed:
                            continue
                        reference_path = sentence_wav(data_root, tgt, reference_sentence)
                        reference_mel = torch.from_numpy(
                            get_mel(reference_path, mel_basis)
                        ).float().unsqueeze(0).to(device)
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
                        seed = (
                            args.seed + 10000 * src_index + 1000 * tgt_index
                            + 100 * source_index + reference_index
                        )
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
                            mel = archived_spectral_subtraction(
                                converted.squeeze(0).cpu().numpy()
                            )
                            audio = hifigan(
                                torch.from_numpy(mel).float().unsqueeze(0).to(device)
                            ).cpu().squeeze().clamp(-1, 1).numpy()
                        output = wav_dir / (
                            f"{variant}__{src}_to_{tgt}__s{source_sentence}"
                            f"__r{reference_sentence}.wav"
                        )
                        sf.write(output, audio, 22050)
                        row = {
                            "variant": variant,
                            "src": src,
                            "tgt": tgt,
                            "source_sentence": source_sentence,
                            "reference_sentence": reference_sentence,
                            "seed": seed,
                            "checkpoint": str(checkpoint),
                            "path": str(output),
                        }
                        append_jsonl(journal, row)
                        completed.add(key)
                        completed_rows.append(row)
                        print(f"GENERATED {len(completed_rows)}/{total} {output.name}", flush=True)
        del model
        torch.cuda.empty_cache()

    if len(completed) != total:
        raise RuntimeError(f"incomplete grid: {len(completed)}/{total}")
    manifest = {"config": config, "rows": completed_rows}
    temporary = output_dir / "generation_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "generation_manifest.json")
    print(f"GENERATION_COMPLETE {total} {output_dir}")


if __name__ == "__main__":
    main()
