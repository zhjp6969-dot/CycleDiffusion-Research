#!/usr/bin/env python3
"""Fail closed unless a clean, independent four-speaker cohort is complete."""

from __future__ import annotations

import argparse
import ast
import json
import struct
from pathlib import Path


DISCOVERY_SPEAKERS = {"p236", "p239", "p259", "p263"}
HELD_OUT_SENTENCES = {"002", "003", "004", "005", "006", "007", "009", "010", "011", "012"}


def strip_suffix(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected filename: {path}")
    return path.name[:-len(suffix)]


def canonical_key(path: Path, suffix: str) -> str:
    key = strip_suffix(path, suffix)
    if key.endswith("_mic1"):
        key = key[:-len("_mic1")]
    return key


def sentence_id(key: str) -> str:
    parts = key.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot parse sentence id from {key!r}")
    return parts[-1]


def npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError("missing NPY magic")
        major, minor = struct.unpack("BB", handle.read(2))
        if (major, minor) == (1, 0):
            header_length = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            header_length = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"unsupported NPY version {(major, minor)}")
        encoding = "utf-8" if major == 3 else "latin1"
        header = ast.literal_eval(handle.read(header_length).decode(encoding).strip())
    shape = header.get("shape")
    if not isinstance(shape, tuple) or not all(isinstance(value, int) for value in shape):
        raise ValueError(f"invalid NPY shape header: {shape!r}")
    return shape


def array_problem(path: Path, kind: str) -> dict | None:
    try:
        shape = npy_shape(path)
    except Exception as exc:  # audit output must retain the corrupt path
        return {"path": str(path), "problem": f"load_error: {exc}"}
    size = 1
    for value in shape:
        size *= value
    if kind == "mel" and (len(shape) != 2 or shape[0] != 80):
        return {"path": str(path), "problem": "invalid_mel_shape", "shape": list(shape)}
    if kind == "embed" and (len(shape) not in (1, 2) or size == 0):
        return {"path": str(path), "problem": "invalid_embedding_shape", "shape": list(shape)}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-size", type=int, default=4)
    parser.add_argument("--minimum-training-items", type=int, default=20)
    args = parser.parse_args()

    roots = {name: args.data_dir / name for name in ("wavs", "mels", "embeds", "txt")}
    missing_roots = sorted(name for name, path in roots.items() if not path.is_dir())
    if missing_roots:
        result = {
            "status": "fail",
            "data_dir": str(args.data_dir),
            "missing_roots": missing_roots,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(1)

    speaker_sets = {
        name: {path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")}
        for name, root in roots.items()
    }
    speakers = set.union(*speaker_sets.values()) if speaker_sets else set()
    overlap = sorted(speakers & DISCOVERY_SPEAKERS)
    rows = []
    failed_speakers = []

    for speaker in sorted(speakers):
        keys = {
            "wavs": {
                canonical_key(path, ".wav")
                for path in (roots["wavs"] / speaker).glob("*_mic1.wav")
            } if (roots["wavs"] / speaker).is_dir() else set(),
            "mels": {
                canonical_key(path, "_mel.npy")
                for path in (roots["mels"] / speaker).glob("*_mel.npy")
            } if (roots["mels"] / speaker).is_dir() else set(),
            "embeds": {
                canonical_key(path, "_embed.npy")
                for path in (roots["embeds"] / speaker).glob("*_embed.npy")
            } if (roots["embeds"] / speaker).is_dir() else set(),
            "txt": {
                path.stem for path in (roots["txt"] / speaker).glob("*.txt")
            } if (roots["txt"] / speaker).is_dir() else set(),
        }
        union_keys = set.union(*keys.values()) if keys else set()
        common_keys = set.intersection(*keys.values()) if keys else set()
        present_sentences = {sentence_id(key) for key in common_keys}
        training_keys = {key for key in common_keys if sentence_id(key) not in HELD_OUT_SENTENCES}
        modality_mismatches = {
            name: sorted(union_keys - values)[:20]
            for name, values in keys.items() if values != union_keys
        }
        array_failures = []
        for path in sorted((roots["mels"] / speaker).glob("*_mel.npy")):
            problem = array_problem(path, "mel")
            if problem:
                array_failures.append(problem)
        for path in sorted((roots["embeds"] / speaker).glob("*_embed.npy")):
            problem = array_problem(path, "embed")
            if problem:
                array_failures.append(problem)
        empty_transcripts = [
            str(path) for path in sorted((roots["txt"] / speaker).glob("*.txt"))
            if not path.read_text(encoding="utf-8", errors="replace").strip()
        ]
        failures = []
        if speaker in DISCOVERY_SPEAKERS:
            failures.append("discovery_speaker_overlap")
        if modality_mismatches:
            failures.append("modality_key_mismatch")
        if len(training_keys) < args.minimum_training_items:
            failures.append("insufficient_training_items")
        if HELD_OUT_SENTENCES - present_sentences:
            failures.append("missing_fixed_evaluation_sentences")
        if array_failures:
            failures.append("invalid_arrays")
        if empty_transcripts:
            failures.append("empty_transcripts")
        if failures:
            failed_speakers.append(speaker)
        rows.append({
            "speaker": speaker,
            "common_items": len(common_keys),
            "training_items_after_exclusion": len(training_keys),
            "missing_fixed_evaluation_sentences": sorted(HELD_OUT_SENTENCES - present_sentences),
            "modality_counts": {name: len(values) for name, values in keys.items()},
            "modality_mismatches_first20": modality_mismatches,
            "array_failures": array_failures,
            "empty_transcripts": empty_transcripts,
            "failures": failures,
        })

    directory_mismatch = any(values != speakers for values in speaker_sets.values())
    status = "pass"
    if len(speakers) != args.cohort_size or overlap or directory_mismatch or failed_speakers:
        status = "fail"
    result = {
        "status": status,
        "data_dir": str(args.data_dir.resolve()),
        "expected_cohort_size": args.cohort_size,
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "discovery_speaker_overlap": overlap,
        "speaker_directories_by_modality": {
            name: sorted(values) for name, values in speaker_sets.items()
        },
        "speaker_directory_mismatch": directory_mismatch,
        "failed_speakers": failed_speakers,
        "minimum_training_items": args.minimum_training_items,
        "held_out_sentences": sorted(HELD_OUT_SENTENCES),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
