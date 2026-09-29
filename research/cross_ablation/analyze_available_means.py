#!/usr/bin/env python3
"""Compute transparent descriptive contrasts from the surviving summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("identity_delta", "robust_content_error", "wer", "cer")
HIGHER_IS_BETTER = {"identity_delta": True, "robust_content_error": False,
                    "wer": False, "cer": False}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer4-summary", type=Path,
        default=Path("data/processed/layer4_full_variant_summary.csv"),
    )
    parser.add_argument(
        "--layer5-summary", type=Path,
        default=Path("data/processed/layer5_lambda_ablation/layer5_variant_summary.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/cross_ablation"),
    )
    return parser.parse_args()


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["variant"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"empty summary: {path}")
    return rows


def main():
    args = parse_args()
    layer4 = load_rows(args.layer4_summary)
    layer5 = load_rows(args.layer5_summary)
    variants = {
        "content_only_e51": layer4["content_only_e51"],
        "joint_e51": layer4["joint_e51"],
        "joint_lambda025_e51": layer5["joint_lambda025_e51"],
    }
    comparisons = (
        ("content_only_minus_joint", "content_only_e51", "joint_e51"),
        (
            "content_only_minus_joint_lambda025",
            "content_only_e51",
            "joint_lambda025_e51",
        ),
    )
    rows = []
    for name, left, right in comparisons:
        for metric in METRICS:
            left_mean = float(variants[left][metric])
            right_mean = float(variants[right][metric])
            difference = left_mean - right_mean
            beneficial = difference > 0 if HIGHER_IS_BETTER[metric] else difference < 0
            rows.append({
                "comparison": name,
                "metric": metric,
                "left_variant": left,
                "right_variant": right,
                "left_mean": left_mean,
                "right_mean": right_mean,
                "mean_difference": difference,
                "direction_favors_left": beneficial,
                "inference_status": "descriptive_only_raw_pairing_unavailable",
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "exploratory_mean_contrasts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "date": "2026-09-22",
        "status": "descriptive_only_blocked_for_paired_inference",
        "comparisons": rows,
        "available_evidence": [
            "Layer 4 variant means",
            "Layer 5 variant means and Layer 5 paired source-cluster rows",
            "rendered Kaggle Version 1 execution record",
        ],
        "missing_evidence": [
            "Layer 4 component-level formal_eval_raw.csv",
            "Layer 4 source-cluster differences for joint_minus_content_only",
        ],
        "recovery_audit": {
            "kaggle_version": 1,
            "script_version_id": 350266488,
            "persisted_output_files": 0,
            "former_draft_archive_bytes": 5486720920,
            "private_asset_component_rows_found": 0,
        },
        "required_next_action": (
            "Restore the original Layer 4 raw table or regenerate only the "
            "content_only_e51 180-item formal grid before paired inference."
        ),
    }
    json_path = args.output_dir / "recovery_audit.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {csv_path}")
    print(f"WROTE {json_path}")
    for row in rows:
        print(row["comparison"], row["metric"], f'{row["mean_difference"]:+.6f}')


if __name__ == "__main__":
    main()
