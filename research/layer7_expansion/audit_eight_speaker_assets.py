#!/usr/bin/env python3
"""Validate an eight-speaker asset tree before Layer 7 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EVAL_SENTENCES = {"002", "003", "004", "005", "006", "007", "009", "010", "011", "012"}


def key_from_path(path, suffix):
    if not path.name.endswith(suffix):
        raise ValueError(path)
    return path.name[:-len(suffix)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-training-items", type=int, default=20)
    args = parser.parse_args()
    roots = {name: args.data_dir / name for name in ("wavs", "mels", "embeds", "txt")}
    missing_roots = [name for name, path in roots.items() if not path.is_dir()]
    if missing_roots:
        raise FileNotFoundError(f"missing asset roots: {missing_roots}")
    speaker_sets = {
        name: {path.name for path in root.iterdir() if path.is_dir()}
        for name, root in roots.items()
    }
    speakers = speaker_sets["mels"]
    if any(value != speakers for value in speaker_sets.values()):
        raise ValueError(f"speaker directory mismatch: {speaker_sets}")
    rows = []
    failures = []
    for speaker in sorted(speakers):
        mel_keys = {
            key_from_path(path, "_mel.npy")
            for path in (roots["mels"] / speaker).glob("*_mel.npy")
        }
        embed_keys = {
            key_from_path(path, "_embed.npy")
            for path in (roots["embeds"] / speaker).glob("*_embed.npy")
        }
        wav_keys = {path.stem for path in (roots["wavs"] / speaker).glob("*_mic1.wav")}
        sentence_ids = {
            key.split("_")[1] for key in mel_keys if len(key.split("_")) > 1
        }
        training_keys = {
            key for key in mel_keys
            if len(key.split("_")) > 1 and key.split("_")[1] not in EVAL_SENTENCES
        }
        shape_failures = []
        for path in sorted((roots["mels"] / speaker).glob("*_mel.npy"))[:20]:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or array.shape[0] != 80:
                shape_failures.append({"path": str(path), "shape": list(array.shape)})
        row = {
            "speaker": speaker,
            "mel_items": len(mel_keys),
            "embed_items": len(embed_keys),
            "wav_items": len(wav_keys),
            "training_items": len(training_keys),
            "missing_eval_sentences": sorted(EVAL_SENTENCES - sentence_ids),
            "mel_without_embedding": sorted(mel_keys - embed_keys)[:20],
            "embedding_without_mel": sorted(embed_keys - mel_keys)[:20],
            "shape_failures_first20": shape_failures,
        }
        if (len(training_keys) < args.minimum_training_items or
                row["missing_eval_sentences"] or row["mel_without_embedding"] or
                row["embedding_without_mel"] or shape_failures):
            failures.append(speaker)
        rows.append(row)
    status = "pass" if len(speakers) == 8 and not failures else "fail"
    result = {
        "status": status,
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "failed_speakers": failures,
        "evaluation_sentences": sorted(EVAL_SENTENCES),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
