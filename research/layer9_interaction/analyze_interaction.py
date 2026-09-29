#!/usr/bin/env python3
"""Analyze the uniform/content-only by cycle-strength 2 x 2 experiment."""

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
VARIANTS = (
    "uniform_lambda100_e51",
    "content_only_lambda100_e51",
    "uniform_lambda025_e51",
    "content_only_lambda025_e51",
)
RUN_DIRS = {
    "content_only_lambda100_e51": "content_only_lambda100_seed37_e51",
    "uniform_lambda025_e51": "uniform_lambda025_seed37_e51",
    "content_only_lambda025_e51": "content_only_lambda025_seed37_e51",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-csv", type=Path, required=True)
    parser.add_argument("--interaction-csv", type=Path, required=True)
    parser.add_argument("--controlled-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def bootstrap_ci(values, generator, repetitions):
    values = np.asarray(values, dtype=np.float64)
    draws = values[generator.integers(
        0, len(values), size=(repetitions, len(values))
    )].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def sign_flip_pvalue(values, generator, repetitions):
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    signs = generator.choice((-1.0, 1.0), size=(repetitions, len(values)))
    randomized = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(randomized >= observed)) / (repetitions + 1))


def training_summary(root, variant):
    if variant == "uniform_lambda100_e51":
        return {
            "variant": variant,
            "status": "reused_formal_evaluation_training_artifacts_not_bundled",
            "steps": None,
            "checkpoint_bytes": None,
        }
    run_dir = root / RUN_DIRS[variant]
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint = run_dir / "vc_51.pt"
    if not metrics_path.exists() or not checkpoint.exists():
        raise FileNotFoundError(f"missing training evidence for {variant}: {run_dir}")
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
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"source_sentence": str, "reference_sentence": str}
    formal = pd.read_csv(args.formal_csv, dtype=dtype)
    formal = formal[formal.variant == "uniform_e51"].copy()
    formal.loc[:, "variant"] = "uniform_lambda100_e51"
    new = pd.read_csv(args.interaction_csv, dtype=dtype)
    frame = pd.concat([formal, new], ignore_index=True)

    expected_per_cell = 4 * 3 * 5 * 3
    expected = len(VARIANTS) * expected_per_cell
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, found {len(frame)}")
    if set(frame.variant) != set(VARIANTS):
        raise ValueError(f"unexpected variants: {sorted(frame.variant.unique())}")
    counts = frame.groupby("variant").size().to_dict()
    if any(counts.get(variant) != expected_per_cell for variant in VARIANTS):
        raise ValueError(f"unbalanced cells: {counts}")
    if frame.duplicated(["variant", *PAIR_FIELDS]).any():
        raise ValueError("duplicate evaluation keys")

    aggregate = frame.groupby("variant", as_index=False)[list(METRICS)].mean()
    aggregate.to_csv(args.output_dir / "layer9_variant_summary.csv", index=False)
    pivot = frame.pivot(index=PAIR_FIELDS, columns="variant", values=list(METRICS))
    if pivot.isna().any().any():
        raise ValueError("paired 2 x 2 grid has missing cells")

    contrasts = {
        "content_minus_uniform_at_lambda100": {
            "content_only_lambda100_e51": 1.0,
            "uniform_lambda100_e51": -1.0,
        },
        "content_minus_uniform_at_lambda025": {
            "content_only_lambda025_e51": 1.0,
            "uniform_lambda025_e51": -1.0,
        },
        "lambda025_minus_lambda100_within_uniform": {
            "uniform_lambda025_e51": 1.0,
            "uniform_lambda100_e51": -1.0,
        },
        "lambda025_minus_lambda100_within_content_only": {
            "content_only_lambda025_e51": 1.0,
            "content_only_lambda100_e51": -1.0,
        },
        "interaction_difference_in_differences": {
            "content_only_lambda025_e51": 1.0,
            "uniform_lambda025_e51": -1.0,
            "content_only_lambda100_e51": -1.0,
            "uniform_lambda100_e51": 1.0,
        },
    }
    generator = np.random.default_rng(args.seed)
    effect_rows, cluster_rows = [], []
    for name, weights in contrasts.items():
        for metric in METRICS:
            differences = sum(
                weight * pivot[(metric, variant)]
                for variant, weight in weights.items()
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
            effect_rows.append({
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

    effects = pd.DataFrame(effect_rows)
    effects.to_csv(args.output_dir / "layer9_paired_effects.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(
        args.output_dir / "layer9_source_cluster_differences.csv", index=False
    )
    training = [training_summary(args.controlled_root, variant) for variant in VARIANTS]
    pd.DataFrame(training).to_csv(
        args.output_dir / "layer9_training_summary.csv", index=False
    )
    primary = effects[
        effects.comparison == "interaction_difference_in_differences"
    ]
    result = {
        "date": "2026-09-22",
        "status": "complete",
        "scope": "four_speakers_uniform_vs_content_only_by_lambda_2x2",
        "evaluation_outputs": len(frame),
        "outputs_per_cell": expected_per_cell,
        "independent_source_clusters": 20,
        "primary_interaction": primary.to_dict("records"),
        "all_effects": effects.to_dict("records"),
        "variant_means": aggregate.to_dict("records"),
        "training": training,
        "inference_warning": (
            "source clusters share four speakers; intervals are paired "
            "cluster-bootstrap summaries and p-values use a Monte Carlo "
            "paired sign-flip test"
        ),
    }
    (args.output_dir / "layer9_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(aggregate.to_string(index=False))
    print(primary.to_string(index=False))
    print(f"LAYER9_ANALYSIS_COMPLETE {args.output_dir}")


if __name__ == "__main__":
    main()
