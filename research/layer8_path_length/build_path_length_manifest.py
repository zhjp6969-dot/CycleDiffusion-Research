#!/usr/bin/env python3
"""Build the frozen one/two/three-hop path-length diagnostic manifest."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path


SPEAKERS = ("p236", "p239", "p259", "p263")
SOURCE_SENTENCES = ("002", "004", "005", "006", "007")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intermediate-reference", default="003")
    parser.add_argument("--anchor-reference", default="009")
    parser.add_argument("--alternate-reference", default="010")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument(
        "--source-sentences",
        default=",".join(SOURCE_SENTENCES),
        help="Comma-separated source sentence IDs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    references = {
        args.intermediate_reference,
        args.anchor_reference,
        args.alternate_reference,
    }
    if len(references) != 3:
        raise ValueError("the three reference sentence IDs must be distinct")
    source_sentences = tuple(
        item.strip() for item in args.source_sentences.split(",") if item.strip()
    )
    if not source_sentences or len(set(source_sentences)) != len(source_sentences):
        raise ValueError("source sentence IDs must be nonempty and unique")

    rows = []
    directions = list(itertools.permutations(SPEAKERS, 2))
    for direction_index, (src, tgt) in enumerate(directions):
        mid_a, mid_b = sorted(set(SPEAKERS) - {src, tgt})
        for source_index, source_sentence in enumerate(source_sentences):
            base = args.seed + 1000 * direction_index + 10 * source_index
            rows.append({
                "unit_id": f"{src}_to_{tgt}__s{source_sentence}",
                "src": src,
                "tgt": tgt,
                "mid_a": mid_a,
                "mid_b": mid_b,
                "source_sentence": source_sentence,
                "intermediate_reference_sentence": args.intermediate_reference,
                "anchor_reference_sentence": args.anchor_reference,
                "alternate_reference_sentence": args.alternate_reference,
                "anchor_seed": base,
                "alternate_seed": base + 1,
                "first_a_seed": base + 2,
                "first_b_seed": base + 3,
                "second_a_b_seed": base + 4,
                "second_b_a_seed": base + 5,
                "final_seed": base,
            })

    expected = len(directions) * len(source_sentences)
    if len(rows) != expected or len({row["unit_id"] for row in rows}) != expected:
        raise RuntimeError("invalid path-length manifest cardinality")
    if expected != 60 and source_sentences == SOURCE_SENTENCES:
        raise RuntimeError(f"frozen manifest must contain 60 units, found {expected}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"PATH_LENGTH_MANIFEST_COMPLETE {len(rows)} {args.output}")


if __name__ == "__main__":
    main()
