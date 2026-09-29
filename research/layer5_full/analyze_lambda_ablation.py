#!/usr/bin/env python3
"""Analyze the fixed joint-cycle lambda ablation on the formal grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_FIELDS = ["src", "tgt", "source_sentence", "reference_sentence"]
METRICS = (
    "identity_delta", "robust_content_error", "wer", "cer",
    "clip_fraction", "silence_fraction", "rms_dbfs", "duration_s",
)


def bootstrap_ci(values, generator, repetitions):
    """Paired percentile interval after collapsing to source-utterance clusters."""
    values = np.asarray(values, dtype=np.float64)
    draws = values[generator.integers(
        0, len(values), size=(repetitions, len(values))
    )].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def sign_flip_pvalue(values, generator, repetitions):
    """Two-sided paired randomization test without an optional SciPy dependency."""
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    signs = generator.choice((-1.0, 1.0), size=(repetitions, len(values)))
    randomized = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(randomized >= observed)) / (repetitions + 1))


VARIANTS = (
    "base_epoch50",
    "joint_lambda100_e51",
    "joint_lambda050_e51",
    "joint_lambda025_e51",
    "joint_lambda000_e51",
)
RUN_DIRS = {
    "joint_lambda100_e51": "joint_seed37_e51",
    "joint_lambda050_e51": "joint_lambda050_seed37_e51",
    "joint_lambda025_e51": "joint_lambda025_seed37_e51",
    "joint_lambda000_e51": "joint_lambda000_seed37_e51",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-csv", type=Path, required=True)
    parser.add_argument("--lambda-csv", type=Path, required=True)
    parser.add_argument("--controlled-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def training_summary(root, variant):
    run_dir = root / RUN_DIRS[variant]
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint = run_dir / "vc_51.pt"
    if not metrics_path.exists():
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
        "status": (
            "training_artifacts_verified" if checkpoint.exists()
            else "training_metrics_verified_checkpoint_not_bundled"
        ),
        "steps": len(rows),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
    }
    for field in (
        "direct_loss", "cycle_loss", "total_loss", "weight_mean",
        "speaker_reliability_mean", "content_reliability_mean",
        "acoustic_reliability_mean", "decoder_grad_norm",
    ):
        values = np.asarray([float(row[field]) for row in rows])
        result[f"{field}_mean"] = float(values.mean())
        result[f"{field}_last25_mean"] = float(values[-25:].mean())
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"source_sentence": str, "reference_sentence": str}
    formal = pd.read_csv(args.formal_csv, dtype=dtype)
    formal = formal[formal.variant.isin(["base_epoch50", "joint_e51"])].copy()
    formal.loc[formal.variant == "joint_e51", "variant"] = "joint_lambda100_e51"
    frame = pd.concat([formal, pd.read_csv(args.lambda_csv, dtype=dtype)], ignore_index=True)
    expected = len(VARIANTS) * 4 * 3 * 5 * 3
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, found {len(frame)}")
    if set(frame.variant) != set(VARIANTS):
        raise ValueError(f"unexpected variants: {sorted(frame.variant.unique())}")
    if frame.duplicated(["variant", *PAIR_FIELDS]).any():
        raise ValueError("duplicate evaluation keys")

    aggregate = frame.groupby("variant", as_index=False)[list(METRICS)].mean()
    aggregate.to_csv(args.output_dir / "layer5_variant_summary.csv", index=False)
    pivot = frame.pivot(index=PAIR_FIELDS, columns="variant", values=list(METRICS))
    if pivot.isna().any().any():
        raise ValueError("paired lambda grid has missing cells")

    comparisons = [
        ("lambda100_minus_lambda000", "joint_lambda100_e51", "joint_lambda000_e51"),
        ("lambda050_minus_lambda000", "joint_lambda050_e51", "joint_lambda000_e51"),
        ("lambda025_minus_lambda000", "joint_lambda025_e51", "joint_lambda000_e51"),
        ("lambda050_minus_lambda100", "joint_lambda050_e51", "joint_lambda100_e51"),
        ("lambda025_minus_lambda100", "joint_lambda025_e51", "joint_lambda100_e51"),
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
            ci_low, ci_high = bootstrap_ci(
                clusters.difference, generator, args.bootstrap_repetitions
            )
            p_sign_flip = sign_flip_pvalue(
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
                "p_sign_flip_mc": p_sign_flip,
                "positive_clusters": int((clusters.difference > 0).sum()),
            })
            clusters.insert(0, "metric", metric)
            clusters.insert(0, "comparison", name)
            cluster_rows.extend(clusters.to_dict("records"))

    comparisons_frame = pd.DataFrame(summary_rows)
    comparisons_frame.to_csv(args.output_dir / "layer5_paired_comparisons.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(
        args.output_dir / "layer5_source_cluster_differences.csv", index=False
    )
    training = [training_summary(args.controlled_root, variant) for variant in VARIANTS[1:]]
    pd.DataFrame(training).to_csv(args.output_dir / "layer5_training_summary.csv", index=False)
    result = {
        "date": "2026-09-22",
        "status": "complete",
        "scope": "four_speakers_joint_lambda_ablation_12_directions_5_sources_3_references",
        "evaluation_outputs": len(frame),
        "independent_source_clusters": 20,
        "lambda_comparisons": comparisons_frame.to_dict("records"),
        "variant_means": aggregate.to_dict("records"),
        "training": training,
        "inference_warning": (
            "source clusters share four speakers; intervals are paired "
            "cluster-bootstrap summaries and p-values use a Monte Carlo "
            "paired sign-flip test"
        ),
    }
    (args.output_dir / "layer5_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(aggregate.to_string(index=False))
    print(comparisons_frame.to_string(index=False))
    print(f"LAYER5_ANALYSIS_COMPLETE {args.output_dir}")


if __name__ == "__main__":
    main()
