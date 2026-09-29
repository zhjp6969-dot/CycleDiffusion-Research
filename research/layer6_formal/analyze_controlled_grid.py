#!/usr/bin/env python3
"""Paired analysis for the completed formal uniform/joint comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon


VARIANTS = ("base_epoch50", "uniform_e51", "joint_e51")
PAIR_FIELDS = ["src", "tgt", "source_sentence", "reference_sentence"]
METRICS = (
    "identity_delta", "robust_content_error", "wer", "cer",
    "clip_fraction", "silence_fraction", "rms_dbfs", "duration_s",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--uniform-run-dir", type=Path, required=True)
    parser.add_argument("--joint-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260915)
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
        w_result = wilcoxon(values)
        p_wilcoxon = float(w_result.pvalue)
    except ValueError:
        p_wilcoxon = 1.0
    return float(t_result.pvalue), p_wilcoxon


def load_training(run_dir, variant):
    metrics_path = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if len(rows) != 461 or [row["global_step"] for row in rows] != list(range(1, 462)):
        raise ValueError(f"{variant} training metrics are not exactly steps 1..461")
    checkpoint = run_dir / "vc_51.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    fields = (
        "direct_loss", "cycle_loss", "total_loss", "weight_mean",
        "speaker_reliability_mean", "content_reliability_mean",
        "acoustic_reliability_mean", "decoder_grad_norm",
    )
    summary = {"variant": variant, "steps": len(rows), "checkpoint_bytes": checkpoint.stat().st_size}
    for field in fields:
        values = [float(row[field]) for row in rows]
        summary[f"{field}_mean"] = float(np.mean(values))
        summary[f"{field}_last25_mean"] = float(np.mean(values[-25:]))
    pid_path = run_dir / "pid.txt"
    if pid_path.exists():
        summary["approx_wall_seconds"] = float(checkpoint.stat().st_mtime - pid_path.stat().st_mtime)
        summary["approx_seconds_per_step"] = summary["approx_wall_seconds"] / len(rows)
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.evaluation_csv, dtype={
        "source_sentence": str, "reference_sentence": str,
    })
    expected = 3 * 4 * 3 * 5 * 3
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, found {len(frame)}")
    if frame.duplicated(["variant", *PAIR_FIELDS]).any():
        raise ValueError("duplicate evaluation keys")
    if set(frame.variant) != set(VARIANTS):
        raise ValueError(f"unexpected variants: {sorted(frame.variant.unique())}")

    aggregate = frame.groupby("variant", as_index=False)[list(METRICS)].mean()
    aggregate.to_csv(args.output_dir / "formal_eval_variant_summary.csv", index=False)
    pivot = frame.pivot(index=PAIR_FIELDS, columns="variant", values=list(METRICS))
    if pivot.isna().any().any():
        raise ValueError("paired evaluation grid has missing cells")

    generator = np.random.default_rng(args.seed)
    comparison_rows = []
    cluster_rows = []
    comparisons = (
        ("joint_minus_uniform", "joint_e51", "uniform_e51"),
        ("uniform_minus_base", "uniform_e51", "base_epoch50"),
        ("joint_minus_base", "joint_e51", "base_epoch50"),
    )
    for name, left, right in comparisons:
        for metric in METRICS:
            differences = (pivot[(metric, left)] - pivot[(metric, right)]).rename("difference").reset_index()
            source_clusters = differences.groupby(
                ["src", "source_sentence"], as_index=False
            ).difference.mean()
            p_t, p_wilcoxon = tests(source_clusters.difference)
            ci = bootstrap_ci(
                source_clusters.difference, generator, args.bootstrap_repetitions
            )
            comparison_rows.append({
                "comparison": name,
                "metric": metric,
                "raw_paired_units": len(differences),
                "source_clusters": len(source_clusters),
                "mean_difference": float(source_clusters.difference.mean()),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "p_t": p_t,
                "p_wilcoxon": p_wilcoxon,
                "positive_clusters": int((source_clusters.difference > 0).sum()),
            })
            source_clusters.insert(0, "metric", metric)
            source_clusters.insert(0, "comparison", name)
            cluster_rows.extend(source_clusters.to_dict("records"))

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.output_dir / "formal_eval_paired_comparisons.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(
        args.output_dir / "formal_eval_source_cluster_differences.csv", index=False
    )
    training = [
        load_training(args.uniform_run_dir, "uniform_e51"),
        load_training(args.joint_run_dir, "joint_e51"),
    ]
    pd.DataFrame(training).to_csv(
        args.output_dir / "formal_training_summary.csv", index=False
    )
    primary = comparison[comparison.comparison == "joint_minus_uniform"]
    summary = {
        "date": "2026-09-15",
        "status": "complete",
        "scope": "four_speakers_12_directions_5_sources_3_references",
        "evaluation_outputs": len(frame),
        "independent_source_clusters": 20,
        "inference_warning": "source clusters share four speakers; intervals are paired cluster bootstrap summaries",
        "data_boundary": "all generation and identity-centroid sentence IDs were excluded from continuation, but the epoch-50 checkpoint may have historical exposure",
        "primary_comparison": primary.to_dict("records"),
        "variant_means": aggregate.to_dict("records"),
        "training": training,
    }
    (args.output_dir / "formal_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(aggregate.to_string(index=False))
    print(primary.to_string(index=False))
    print(pd.DataFrame(training).to_string(index=False))
    print(f"ANALYSIS_COMPLETE {args.output_dir}")


if __name__ == "__main__":
    main()
