#!/usr/bin/env python3
"""Evaluate the formal uniform/joint grid with disjoint automatic metrics."""

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


SPEAKERS = ("p236", "p239", "p259", "p263")
EVAL_CENTROID_SENTENCES = ("011", "012")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
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


def key(row):
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.project_root / "VCTK_2F2M"
    manifest = json.loads(
        (args.generated_dir / "generation_manifest.json").read_text()
    )
    rows = manifest["rows"]
    expected = int(manifest["config"]["expected_outputs"])
    if len(rows) != expected or len({key(row) for row in rows}) != expected:
        raise ValueError(
            f"generation manifest is not the complete {expected}-item grid"
        )
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

    missing = [row for row in rows if key(row) not in completed]
    asr_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
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
        return {"text": asr_processor.batch_decode(
            token_ids, skip_special_tokens=True
        )[0]}

    for start in range(0, len(missing), args.batch_size):
        batch = missing[start:start + args.batch_size]
        # Direct processor/model inference avoids a Transformers pipeline
        # incompatibility where Whisper preprocessing expects a missing
        # ``num_frames`` field.
        hypotheses = [transcribe(row["path"]) for row in batch]
        for row, transcription in zip(batch, hypotheses):
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
            result = {
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
            }
            append_jsonl(journal, result)
            completed.add(key(result))
            evaluated.append(result)
            print(f"EVALUATED {len(evaluated)}/{expected} {path.name}", flush=True)

    if len(completed) != expected:
        raise RuntimeError(f"incomplete evaluation: {len(completed)}/{expected}")
    frame = pd.DataFrame(evaluated).sort_values([
        "variant", "src", "tgt", "source_sentence", "reference_sentence"
    ])
    temporary = args.output_dir / "formal_eval_raw.csv.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, args.output_dir / "formal_eval_raw.csv")
    print(f"EVALUATION_COMPLETE {expected} {args.output_dir}")


if __name__ == "__main__":
    main()
