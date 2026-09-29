#!/usr/bin/env python3
"""Create the four aligned asset modalities for a new VCTK cohort.

Heavy imports are intentionally delayed until after argument validation so that
the script can be syntax-checked in a lightweight repository environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path


DISCOVERY_SPEAKERS = {"p236", "p239", "p259", "p263"}
REQUIRED_SENTENCES = {"002", "003", "004", "005", "006", "007", "009", "010", "011", "012"}
AUDIO_SUFFIXES = {".wav", ".flac"}
KEY_PATTERN = re.compile(r"^(p\d+)_(\d{3})(?:_mic1)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_audio_key(path: Path) -> tuple[str, str] | None:
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        return None
    stem = path.stem
    if stem.endswith("_mic2"):
        return None
    match = KEY_PATTERN.match(stem)
    if not match:
        return None
    return match.group(1), match.group(2)


def find_audio(raw_root: Path, speaker: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(raw_root.rglob(f"{speaker}_*")):
        parsed = canonical_audio_key(path)
        if parsed is None or parsed[0] != speaker:
            continue
        sentence = parsed[1]
        incumbent = found.get(sentence)
        if incumbent is None or "_mic1" in path.stem:
            found[sentence] = path
    return found


def find_transcripts(raw_root: Path, speaker: str) -> dict[str, Path]:
    found = {}
    for path in sorted(raw_root.rglob(f"{speaker}_*.txt")):
        match = KEY_PATTERN.match(path.stem)
        if match and match.group(1) == speaker:
            found[match.group(2)] = path
    return found


def exact_mel(wav, librosa, np, mel_basis):
    wav = wav[: (wav.shape[0] // 256) * 256]
    if wav.size < 2:
        raise ValueError("waveform is too short")
    wav = np.pad(wav, 384, mode="reflect")
    stft = librosa.core.stft(
        wav, n_fft=1024, hop_length=256, win_length=1024,
        window="hann", center=False,
    )
    magnitude = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
    mel = np.matmul(mel_basis, magnitude)
    return np.log(np.clip(mel, a_min=1e-5, a_max=None)).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-wav-root", type=Path, required=True)
    parser.add_argument("--raw-txt-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speakers", nargs=4, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--limit-per-speaker", type=int, default=0)
    args = parser.parse_args()

    speakers = tuple(args.speakers)
    if len(set(speakers)) != 4:
        raise ValueError(f"four distinct speakers are required: {speakers}")
    overlap = sorted(set(speakers) & DISCOVERY_SPEAKERS)
    if overlap:
        raise ValueError(f"discovery speakers are forbidden in Layer 10: {overlap}")
    if not args.raw_wav_root.is_dir() or not args.raw_txt_root.is_dir():
        raise FileNotFoundError("raw waveform and transcript roots must both exist")

    checkpoint = args.encoder_checkpoint
    if checkpoint is None:
        checkpoint = (
            args.project_root / "checkpts" / "spk_encoder" / "pretrained.pt"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "speaker encoder pretrained.pt not found; pass the archived speaker "
            f"encoder explicitly with --encoder-checkpoint: {checkpoint}"
        )
    if checkpoint.name == "enc.pt":
        raise ValueError(
            "enc.pt is the CycleDiffusion acoustic encoder state dict, not the "
            "speaker encoder checkpoint; use the archived pretrained.pt"
        )
    encoder_root = args.project_root / "speaker_encoder"
    if not encoder_root.is_dir():
        raise FileNotFoundError(f"speaker encoder source not found: {encoder_root}")

    import librosa
    import numpy as np
    import soundfile as sf
    from librosa.filters import mel as librosa_mel_fn

    sys.path.insert(0, str(encoder_root))
    from encoder import inference as spk_encoder

    mel_basis = librosa_mel_fn(
        sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000
    )
    spk_encoder.load_model(checkpoint, device="cpu")

    roots = {name: args.output_dir / name for name in ("wavs", "mels", "embeds", "txt")}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "preprocess_progress.jsonl"
    completed = set()
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row["speaker"], row["sentence"]))

    rows = []
    for speaker in speakers:
        audio = find_audio(args.raw_wav_root, speaker)
        transcripts = find_transcripts(args.raw_txt_root, speaker)
        missing_required = sorted(REQUIRED_SENTENCES - (set(audio) & set(transcripts)))
        if missing_required:
            raise ValueError(f"{speaker} lacks required sentences: {missing_required}")
        sentences = sorted(set(audio) & set(transcripts))
        if args.limit_per_speaker:
            sentences = sentences[: args.limit_per_speaker]
        for sentence in sentences:
            source_wav = audio[sentence]
            source_txt = transcripts[sentence]
            key = f"{speaker}_{sentence}_mic1"
            wav_out = roots["wavs"] / speaker / f"{key}.wav"
            mel_out = roots["mels"] / speaker / f"{key}_mel.npy"
            embed_out = roots["embeds"] / speaker / f"{key}_embed.npy"
            txt_out = roots["txt"] / speaker / f"{speaker}_{sentence}.txt"
            for path in (wav_out, mel_out, embed_out, txt_out):
                path.parent.mkdir(parents=True, exist_ok=True)

            item_key = (speaker, sentence)
            outputs_exist = all(path.is_file() for path in (wav_out, mel_out, embed_out, txt_out))
            if item_key not in completed or not outputs_exist:
                wav, _ = librosa.load(source_wav, sr=22050, mono=True)
                sf.write(wav_out, wav, 22050, subtype="PCM_16")
                mel = exact_mel(wav, librosa, np, mel_basis)
                np.save(mel_out, mel)
                encoder_wav = spk_encoder.preprocess_wav(str(wav_out))
                embedding = spk_encoder.embed_utterance(encoder_wav).astype("float32")
                np.save(embed_out, embedding)
                shutil.copy2(source_txt, txt_out)
                record = {
                    "speaker": speaker,
                    "sentence": sentence,
                    "source_wav": str(source_wav),
                    "source_wav_sha256": sha256(source_wav),
                    "wav": str(wav_out),
                    "wav_sha256": sha256(wav_out),
                    "mel": str(mel_out),
                    "mel_shape": list(mel.shape),
                    "embed": str(embed_out),
                    "embed_shape": list(embedding.shape),
                    "txt": str(txt_out),
                }
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                completed.add(item_key)
            rows.append({
                "speaker": speaker,
                "sentence": sentence,
                "source_wav": str(source_wav),
                "wav": str(wav_out),
                "mel": str(mel_out),
                "embed": str(embed_out),
                "txt": str(txt_out),
            })

    manifest = args.output_dir / "preprocess_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "speakers": list(speakers),
        "items": len(rows),
        "items_by_speaker": {
            speaker: sum(row["speaker"] == speaker for row in rows)
            for speaker in speakers
        },
        "sample_rate": 22050,
        "mel_bins": 80,
        "speaker_encoder_checkpoint": str(checkpoint),
        "speaker_encoder_checkpoint_sha256": sha256(checkpoint),
        "manifest": str(manifest),
    }
    (args.output_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
