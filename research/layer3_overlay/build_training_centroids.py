#!/usr/bin/env python3
"""Build per-speaker centroids from training-reference embedding files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-sentences", default="002,003,004,005,006,007,009,010,011,012")
    args = parser.parse_args()
    excluded = set(filter(None, args.exclude_sentences.split(",")))
    centroids = {}
    for speaker_dir in sorted(path for path in args.embed_dir.iterdir() if path.is_dir()):
        vectors = []
        for path in sorted(speaker_dir.glob("*_embed.npy")):
            fields = path.stem.split("_")
            if len(fields) > 1 and fields[1] in excluded:
                continue
            vectors.append(np.load(path).reshape(-1))
        if vectors:
            centroid = np.mean(np.stack(vectors), axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            centroids[speaker_dir.name] = centroid.astype(np.float32)
    if not centroids:
        raise RuntimeError("no training embeddings found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **centroids)
    print(f"saved {len(centroids)} speaker centroids to {args.output}")


if __name__ == "__main__":
    main()
