#!/usr/bin/env python3
"""Descriptive non-circular evaluation for the fixed checkpoint pilot grid."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector, pipeline


SPEAKERS = ["p236", "p239", "p259", "p263"]
EVAL_CENTROID_SENTENCES = ["004", "005", "006", "007", "009", "010", "011", "012"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-8))


def bootstrap_ci(values, seed=20260915, repetitions=50000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = values[generator.integers(0, len(values), size=(repetitions, len(values)))]
    return [float(value) for value in np.quantile(samples.mean(axis=1), [0.025, 0.975])]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.project_root / "VCTK_2F2M"
    manifest = json.loads(
        (args.generated_dir / "generation_manifest.json").read_text()
    )
    rows = manifest["rows"]
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
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding.cpu().numpy()

    centroids = {}
    for speaker in SPEAKERS:
        embeddings = []
        for sentence in EVAL_CENTROID_SENTENCES:
            path = data_root / "wavs" / speaker / f"{speaker}_{sentence}_mic1.wav"
            embeddings.append(identity_embedding(path))
        centroid = np.mean(np.stack(embeddings), axis=0)
        centroids[speaker] = centroid / (np.linalg.norm(centroid) + 1e-8)

    audio_paths = [row["path"] for row in rows]
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base.en",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    hypotheses = asr(audio_paths, batch_size=8)
    evaluated = []
    for row, transcription in zip(rows, hypotheses):
        path = Path(row["path"])
        embedding = identity_embedding(path)
        identity_delta = cosine(embedding, centroids[row["tgt"]])
        identity_delta -= cosine(embedding, centroids[row["src"]])
        text_path = (
            data_root / "txt" / row["src"] /
            f"{row['src']}_{row['source_sentence']}.txt"
        )
        reference = text_path.read_text(errors="ignore").strip()
        hypothesis = transcription["text"]
        wer, cer, robust_error = content_scores(reference, hypothesis)
        audio, rate = load_audio(path)
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        evaluated.append({
            **row,
            "identity_delta": identity_delta,
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
        })
    frame = pd.DataFrame(evaluated)
    frame.to_csv(args.output_dir / "layer6_pilot_raw.csv", index=False)

    metric_names = [
        "identity_delta", "robust_content_error", "wer", "cer",
        "duration_s", "rms_dbfs", "clip_fraction", "silence_fraction",
    ]
    aggregate = frame.groupby("variant", as_index=False)[metric_names].mean()
    aggregate.to_csv(args.output_dir / "layer6_pilot_summary.csv", index=False)
    base = frame[frame.variant == "base_epoch50"].sort_values(["src", "tgt"])
    comparisons = []
    for variant in ["uniform_pilot20", "joint_pilot20"]:
        current = frame[frame.variant == variant].sort_values(["src", "tgt"])
        for metric in ["identity_delta", "robust_content_error", "clip_fraction"]:
            differences = current[metric].to_numpy() - base[metric].to_numpy()
            comparisons.append({
                "variant": variant,
                "metric": metric,
                "mean_change_vs_base": float(differences.mean()),
                "descriptive_direction_bootstrap_ci": bootstrap_ci(differences),
            })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(
        args.output_dir / "layer6_pilot_vs_base.csv", index=False
    )
    result = {
        "date": "2026-09-15",
        "scope": "descriptive_12_direction_single_source_single_reference_pilot",
        "inference_warning": "directions share speakers; bootstrap intervals are descriptive, not confirmatory",
        "data_boundary": "evaluation sentence IDs excluded from continuation, but epoch-50 checkpoint history may include them",
        "identity_model": "microsoft/wavlm-base-plus-sv",
        "content_model": "openai/whisper-base.en",
        "identity_centroid_sentences": EVAL_CENTROID_SENTENCES,
        "aggregate": aggregate.to_dict("records"),
        "comparisons": comparisons,
    }
    (args.output_dir / "layer6_pilot_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(aggregate.to_string(index=False))
    print(comparison_frame.to_string(index=False))
    print(f"saved Layer 6 pilot evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
