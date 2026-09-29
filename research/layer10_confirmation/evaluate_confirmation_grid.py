#!/usr/bin/env python3
"""Evaluate the complete Layer 10 grid with disjoint automatic metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioXVector,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


KEY_FIELDS = (
    "seed", "variant", "source_speaker", "target_speaker",
    "source_sentence", "reference_sentence",
)
CENTROID_SENTENCES = ("011", "012")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def key(row):
    return tuple(str(row[field]) for field in KEY_FIELDS)


def load_audio(path, sample_rate=None):
    audio, rate = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sample_rate is not None and rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sample_rate)
        rate = sample_rate
    return audio, rate


def normalize_text(text):
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(text).lower()))


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for column, right_item in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def content_scores(reference, hypothesis):
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    reference_words, hypothesis_words = reference.split(), hypothesis.split()
    word_edits = edit_distance(reference_words, hypothesis_words)
    reference_chars = reference.replace(" ", "")
    hypothesis_chars = hypothesis.replace(" ", "")
    char_edits = edit_distance(reference_chars, hypothesis_chars)
    wer = word_edits / max(1, len(reference_words))
    cer = char_edits / max(1, len(reference_chars))
    return wer, cer, max(min(wer, 1.0), min(cer, 1.0))


def cosine(left, right):
    return float(np.dot(left, right) / (
        np.linalg.norm(left) * np.linalg.norm(right) + 1e-8
    ))


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (args.generated_dir / "generation_manifest.json").read_text()
    )
    rows = manifest["rows"]
    if len(rows) != 1080 or len({key(row) for row in rows}) != 1080:
        raise ValueError("generation manifest is not the complete 1,080-item grid")
    speakers = sorted({row["source_speaker"] for row in rows})
    if len(speakers) != 4:
        raise ValueError(f"expected four speakers, found {speakers}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    identity_model = AutoModelForAudioXVector.from_pretrained(
        "microsoft/wavlm-base-plus-sv"
    ).to(device).eval()

    @torch.inference_mode()
    def identity_embedding(path):
        audio, _ = load_audio(path, 16000)
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        embedding = identity_model(**inputs).embeddings[0]
        return torch.nn.functional.normalize(embedding, dim=-1).cpu().numpy()

    centroids = {}
    for speaker in speakers:
        embeddings = [
            identity_embedding(
                args.data_root / "wavs" / speaker /
                f"{speaker}_{sentence}_mic1.wav"
            )
            for sentence in CENTROID_SENTENCES
        ]
        centroid = np.mean(np.stack(embeddings), axis=0)
        centroids[speaker] = centroid / (np.linalg.norm(centroid) + 1e-8)
        print(f"CENTROID_READY {speaker}", flush=True)

    journal = args.output_dir / "formal_eval_progress.jsonl"
    evaluated = []
    completed = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_key = key(row)
            if row_key in completed:
                raise ValueError(f"duplicate evaluation key: {row_key}")
            completed.add(row_key)
            evaluated.append(row)

    asr_dtype = torch.float16 if device.type == "cuda" else torch.float32
    asr_processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-base.en", torch_dtype=asr_dtype,
    ).to(device).eval()

    @torch.inference_mode()
    def transcribe(path):
        audio, _ = load_audio(path, 16000)
        features = asr_processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device=device, dtype=asr_dtype)
        token_ids = asr_model.generate(features)
        return asr_processor.batch_decode(token_ids, skip_special_tokens=True)[0]

    for row in rows:
        if key(row) in completed:
            continue
        path = Path(row["path"])
        if not path.exists():
            path = args.generated_dir / "audio" / path.name
        if not path.is_file():
            raise FileNotFoundError(path)
        embedding = identity_embedding(path)
        target_cosine = cosine(embedding, centroids[row["target_speaker"]])
        source_cosine = cosine(embedding, centroids[row["source_speaker"]])
        text_path = (
            args.data_root / "txt" / row["source_speaker"] /
            f"{row['source_speaker']}_{row['source_sentence']}.txt"
        )
        reference = text_path.read_text(errors="ignore").strip()
        hypothesis = transcribe(path)
        wer, cer, robust_error = content_scores(reference, hypothesis)
        audio, rate = load_audio(path)
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        result = {
            **row,
            "path": str(path),
            "target_cosine": target_cosine,
            "source_cosine": source_cosine,
            "identity_delta": target_cosine - source_cosine,
            "reference_text": reference,
            "hypothesis": hypothesis,
            "wer": wer,
            "cer": cer,
            "robust_content_error": robust_error,
            "duration_s": len(audio) / rate,
            "rms_dbfs": 20 * np.log10(rms + 1e-12),
            "peak_dbfs": 20 * np.log10(float(np.max(np.abs(audio))) + 1e-12),
            "clip_fraction": float(np.mean(np.abs(audio) >= 0.999)),
            "silence_fraction": float(np.mean(np.abs(audio) < 10 ** (-50 / 20))),
        }
        append_jsonl(journal, result)
        completed.add(key(result))
        evaluated.append(result)
        print(f"EVALUATED {len(evaluated)}/1080 {path.name}", flush=True)

    frame = pd.DataFrame(evaluated).sort_values(list(KEY_FIELDS))
    temporary = args.output_dir / "formal_eval_raw.csv.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, args.output_dir / "formal_eval_raw.csv")
    print(f"LAYER10_EVALUATION_COMPLETE 1080 {args.output_dir}")


if __name__ == "__main__":
    main()
