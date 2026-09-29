#!/usr/bin/env python3
"""Preregistered seed-aware analysis for the Layer 10 confirmation grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (17, 37, 73)
VARIANTS = ("uniform_lambda100", "content_only_lambda100")
UNIT_FIELDS = ("seed", "source_speaker", "source_sentence")
PAIR_FIELDS = (
    "seed", "source_speaker", "target_speaker",
    "source_sentence", "reference_sentence",
)
METRICS = (
    "identity_delta", "robust_content_error", "wer", "cer",
    "clip_fraction", "silence_fraction", "rms_dbfs", "peak_dbfs",
    "duration_s", "generation_seconds",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    return parser.parse_args()


def hierarchical_bootstrap(unit_frame, metric, generator, repetitions):
    by_seed = {
        seed: group[metric].to_numpy(dtype=np.float64)
        for seed, group in unit_frame.groupby("seed")
    }
    seed_values = np.array(sorted(by_seed))
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled_seeds = generator.choice(seed_values, size=len(seed_values), replace=True)
        sampled_means = []
        for seed in sampled_seeds:
            values = by_seed[int(seed)]
            sampled_means.append(generator.choice(
                values, size=len(values), replace=True
            ).mean())
        draws[index] = np.mean(sampled_means)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.evaluation_csv, dtype={
        "source_sentence": str, "reference_sentence": str,
    })
    if len(frame) != 1080:
        raise ValueError(f"expected 1080 rows, found {len(frame)}")
    if frame.duplicated(["variant", *PAIR_FIELDS]).any():
        raise ValueError("duplicate evaluation keys")
    if set(frame.seed) != set(SEEDS) or set(frame.variant) != set(VARIANTS):
        raise ValueError("unexpected seeds or variants")

    available_metrics = [metric for metric in METRICS if metric in frame.columns]
    variant_summary = frame.groupby(
        ["seed", "variant"], as_index=False
    )[available_metrics].mean()
    variant_summary.to_csv(
        args.output_dir / "layer10_seed_variant_summary.csv", index=False
    )

    paired = frame.pivot(
        index=list(PAIR_FIELDS), columns="variant", values=available_metrics
    )
    if paired.isna().any().any():
        raise ValueError("paired Layer 10 grid has missing cells")

    unit_rows = []
    for metric in available_metrics:
        difference = (
            paired[(metric, "content_only_lambda100")]
            - paired[(metric, "uniform_lambda100")]
        ).rename("difference").reset_index()
        collapsed = difference.groupby(
            list(UNIT_FIELDS), as_index=False
        ).difference.mean()
        collapsed.insert(3, "metric", metric)
        unit_rows.extend(collapsed.to_dict("records"))
    unit_frame = pd.DataFrame(unit_rows)
    unit_frame.to_csv(
        args.output_dir / "layer10_source_unit_differences.csv", index=False
    )

    generator = np.random.default_rng(args.bootstrap_seed)
    pooled_rows = []
    seed_rows = []
    speaker_rows = []
    for metric in available_metrics:
        metric_units = unit_frame[unit_frame.metric == metric]
        ci_low, ci_high = hierarchical_bootstrap(
            metric_units, "difference", generator, args.bootstrap_repetitions
        )
        seed_means = metric_units.groupby("seed").difference.mean()
        pooled_rows.append({
            "comparison": "content_only_minus_uniform",
            "metric": metric,
            "source_units": len(metric_units),
            "seeds": metric_units.seed.nunique(),
            "mean_difference": float(metric_units.difference.mean()),
            "hierarchical_ci_low": ci_low,
            "hierarchical_ci_high": ci_high,
            "positive_seed_count": int((seed_means > 0).sum()),
        })
        for seed, group in metric_units.groupby("seed"):
            seed_rows.append({
                "metric": metric,
                "seed": int(seed),
                "source_units": len(group),
                "mean_difference": float(group.difference.mean()),
                "positive_source_units": int((group.difference > 0).sum()),
            })
        for speaker, group in metric_units.groupby("source_speaker"):
            speaker_rows.append({
                "metric": metric,
                "source_speaker": speaker,
                "seed_source_units": len(group),
                "mean_difference": float(group.difference.mean()),
            })

    pooled = pd.DataFrame(pooled_rows)
    per_seed = pd.DataFrame(seed_rows)
    per_speaker = pd.DataFrame(speaker_rows)
    pooled.to_csv(args.output_dir / "layer10_pooled_effects.csv", index=False)
    per_seed.to_csv(args.output_dir / "layer10_per_seed_effects.csv", index=False)
    per_speaker.to_csv(args.output_dir / "layer10_per_speaker_effects.csv", index=False)

    identity = pooled[pooled.metric == "identity_delta"].iloc[0]
    content = pooled[pooled.metric == "robust_content_error"].iloc[0]
    identity_promoted = bool(
        identity.positive_seed_count >= 2
        and identity.hierarchical_ci_low > 0
    )
    summary = {
        "status": "complete",
        "scope": "independent_four_speaker_cohort_three_seeds",
        "evaluation_outputs": len(frame),
        "analysis_units": 60,
        "analysis_unit_definition": (
            "seed/source_speaker/source_sentence after averaging target and reference repeats"
        ),
        "comparison": "content_only_lambda100_minus_uniform_lambda100",
        "identity_promotion_criterion_met": identity_promoted,
        "content_guard": {
            "mean_difference": float(content.mean_difference),
            "hierarchical_ci": [
                float(content.hierarchical_ci_low),
                float(content.hierarchical_ci_high),
            ],
            "adjudication": (
                "report_only: the frozen protocol did not assign a numeric "
                "material-penalty margin"
            ),
        },
        "automatic_promotion": False,
        "automatic_promotion_reason": (
            "identity is rule-based; content and acoustic guards require the "
            "predeclared qualitative review because no numeric margins were frozen"
        ),
        "pooled_effects": pooled.to_dict("records"),
        "per_seed_effects": per_seed.to_dict("records"),
    }
    (args.output_dir / "layer10_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(pooled.to_string(index=False))
    print(per_seed.to_string(index=False))
    print(f"LAYER10_ANALYSIS_COMPLETE {args.output_dir}")


if __name__ == "__main__":
    main()
