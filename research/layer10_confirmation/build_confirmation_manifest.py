#!/usr/bin/env python3
"""Build the frozen 1,080-row Layer 10 evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import permutations
from pathlib import Path


SEEDS = (17, 37, 73)
VARIANTS = ("uniform_lambda100", "content_only_lambda100")
SOURCE_SENTENCES = ("002", "004", "005", "006", "007")
REFERENCE_SENTENCES = ("003", "009", "010")
CENTROID_SENTENCES = ("011", "012")


def paired_generation_seed(seed: int, source: str, target: str, sentence: str, reference: str) -> int:
    payload = f"layer10|{seed}|{source}|{target}|{sentence}|{reference}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    if audit.get("status") != "pass":
        raise ValueError("asset audit did not pass")
    speakers = tuple(audit.get("speakers", ()))
    if len(speakers) != 4:
        raise ValueError(f"expected four replication speakers, got {speakers}")

    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            for source, target in permutations(speakers, 2):
                for sentence in SOURCE_SENTENCES:
                    for reference in REFERENCE_SENTENCES:
                        rows.append({
                            "seed": seed,
                            "variant": variant,
                            "source_speaker": source,
                            "target_speaker": target,
                            "source_sentence": sentence,
                            "reference_sentence": reference,
                            "centroid_sentences": ";".join(CENTROID_SENTENCES),
                            "generation_seed": paired_generation_seed(
                                seed, source, target, sentence, reference
                            ),
                        })
    if len(rows) != 1080:
        raise AssertionError(f"expected 1080 rows, got {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "ready",
        "speakers": list(speakers),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "directions": len(speakers) * (len(speakers) - 1),
        "source_sentences": list(SOURCE_SENTENCES),
        "reference_sentences": list(REFERENCE_SENTENCES),
        "centroid_sentences": list(CENTROID_SENTENCES),
        "rows": len(rows),
        "outputs_per_seed_variant": 180,
        "paired_seed_rule": "sha256(layer10|seed|source|target|sentence|reference) first 31 bits",
        "audit": str(args.audit),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
