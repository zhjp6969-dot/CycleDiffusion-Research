#!/usr/bin/env python3
"""Analyze the complete reliability-component ablation on one fixed grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon


VARIANTS = (
    "base_epoch50", "uniform_e51", "speaker_only_e51",
    "content_only_e51", "joint_e51", "hard_gate_e51",
)
PAIR_FIELDS = ["src", "tgt", "source_sentence", "reference_sentence"]
METRICS = (
    "identity_delta", "robust_content_error", "wer", "cer",
    "clip_fraction", "silence_fraction", "rms_dbfs", "duration_s",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-csv", type=Path, required=True)
    parser.add_argument("--component-csv", type=Path, required=True)
    parser.add_argument("--controlled-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def bootstrap_ci(values, generator, repetitions):
    values = np.asarray(values, dtype=np.float64)
    draws = values[generator.integers(
        0, len(values), size=(repetitions, len(values))
    )].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def tests(values):
    values = np.asarray(values, dtype=np.float64)
    t_result = ttest_1samp(values, 0.0)
    try:
        p_wilcoxon = float(wilcoxon(values).pvalue)
    except ValueError:
        p_wilcoxon = 1.0
    return float(t_result.pvalue), p_wilcoxon


def training_summary(root, variant):
    run_dir = root / f"{variant.removesuffix('_e51')}_seed37_e51"
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint = run_dir / "vc_51.pt"
    # Uniform and joint can be reused from the earlier formal evaluation
    # without their large training artifacts being copied into the Kaggle
    # bundle.  Keep that provenance explicit instead of failing analysis.
    if not metrics_path.exists() or not checkpoint.exists():
        return {
            "variant": variant,
            "status": "reused_evaluation_training_artifacts_not_bundled",
            "steps": None,
            "checkpoint_bytes": None,
        }
    rows = [
        json.loads(line) for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 461 or [row["global_step"] for row in rows] != list(range(1, 462)):
        raise ValueError(f"{variant} training metrics are not steps 1..461")
    result = {
        "variant": variant,
        "status": "training_artifacts_verified",
        "steps": len(rows),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    for field in (
        "direct_loss", "cycle_loss", "total_loss", "weight_mean",
        "speaker_reliability_mean", "content_reliability_mean",
        "acoustic_reliability_mean", "decoder_grad_norm",
    ):
        values = np.asarray([float(row[field]) for row in rows])
        result[f"{field}_mean"] = float(values.mean())
        result[f"{field}_last25_mean"] = float(values[-25:].mean())
    pid = run_dir / "pid.txt"
    if pid.exists():
        result["approx_wall_seconds"] = float(checkpoint.stat().st_mtime - pid.stat().st_mtime)
        result["approx_seconds_per_step"] = result["approx_wall_seconds"] / len(rows)
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([
        pd.read_csv(args.formal_csv, dtype={
            "source_sentence": str, "reference_sentence": str,
        }),
        pd.read_csv(args.component_csv, dtype={
            "source_sentence": str, "reference_sentence": str,
        }),
    ], ignore_index=True)
    expected = len(VARIANTS) * 4 * 3 * 5 * 3
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, found {len(frame)}")
    if set(frame.variant) != set(VARIANTS):
        raise ValueError(f"unexpected variants: {sorted(frame.variant.unique())}")
    if frame.duplicated(["variant", *PAIR_FIELDS]).any():
        raise ValueError("duplicate evaluation keys")

    aggregate = frame.groupby("variant", as_index=False)[list(METRICS)].mean()
    aggregate.to_csv(args.output_dir / "layer4_variant_summary.csv", index=False)
    pivot = frame.pivot(index=PAIR_FIELDS, columns="variant", values=list(METRICS))
    if pivot.isna().any().any():
        raise ValueError("paired grid has missing cells")

    comparisons = [
        (f"{variant}_minus_uniform", variant, "uniform_e51")
        for variant in ("speaker_only_e51", "content_only_e51", "joint_e51", "hard_gate_e51")
    ] + [
        ("joint_minus_speaker_only", "joint_e51", "speaker_only_e51"),
        ("joint_minus_content_only", "joint_e51", "content_only_e51"),
        ("joint_minus_hard_gate", "joint_e51", "hard_gate_e51"),
    ]
    generator = np.random.default_rng(args.seed)
    summary_rows, cluster_rows = [], []
    for name, left, right in comparisons:
        for metric in METRICS:
            differences = (
                pivot[(metric, left)] - pivot[(metric, right)]
            ).rename("difference").reset_index()
            clusters = differences.groupby(
                ["src", "source_sentence"], as_index=False
            ).difference.mean()
            p_t, p_wilcoxon = tests(clusters.difference)
            ci_low, ci_high = bootstrap_ci(
                clusters.difference, generator, args.bootstrap_repetitions
            )
            summary_rows.append({
                "comparison": name,
                "metric": metric,
                "raw_paired_units": len(differences),
                "source_clusters": len(clusters),
                "mean_difference": float(clusters.difference.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_t": p_t,
                "p_wilcoxon": p_wilcoxon,
                "positive_clusters": int((clusters.difference > 0).sum()),
            })
            clusters.insert(0, "metric", metric)
            clusters.insert(0, "comparison", name)
            cluster_rows.extend(clusters.to_dict("records"))

    comparisons_frame = pd.DataFrame(summary_rows)
    comparisons_frame.to_csv(
        args.output_dir / "layer4_paired_comparisons.csv", index=False
    )
    pd.DataFrame(cluster_rows).to_csv(
        args.output_dir / "layer4_source_cluster_differences.csv", index=False
    )
    training = [
        training_summary(args.controlled_root, variant)
        for variant in VARIANTS if variant != "base_epoch50"
    ]
    pd.DataFrame(training).to_csv(
        args.output_dir / "layer4_training_summary.csv", index=False
    )
    primary = comparisons_frame[
        comparisons_frame.comparison.str.endswith("_minus_uniform")
    ]
    result = {
        "date": "2026-09-16",
        "status": "complete",
        "scope": "four_speakers_12_directions_5_sources_3_references",
        "evaluation_outputs": len(frame),
        "independent_source_clusters": 20,
        "primary_component_comparisons": primary.to_dict("records"),
        "variant_means": aggregate.to_dict("records"),
        "training": training,
        "inference_warning": (
            "source clusters share four speakers; intervals are paired "
            "cluster-bootstrap summaries"
        ),
    }
    (args.output_dir / "layer4_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(aggregate.to_string(index=False))
    print(primary.to_string(index=False))
    print(f"LAYER4_ANALYSIS_COMPLETE {args.output_dir}")


if __name__ == "__main__":
    main()
