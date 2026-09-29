#!/usr/bin/env python3
"""Analyze the frozen one/two/three-hop path-length diagnostic."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROLES = (
    "anchor",
    "stochastic",
    "alternate_reference",
    "two_hop_via_a",
    "two_hop_via_b",
    "three_hop_via_a_b",
    "three_hop_via_b_a",
)
PAIR_ROLES = ROLES[1:]
UNIT_FIELDS = [
    "variant", "unit_id", "src", "tgt", "mid_a", "mid_b", "source_sentence"
]
CLUSTER_FIELDS = ["src", "source_sentence"]
METRICS = (
    "wavlm_cosine_distance",
    "word_edit_distance",
    "char_edit_distance",
    "identity_delta_change",
    "identity_delta_abs_change",
    "robust_content_error_change",
    "robust_content_error_abs_change",
    "silence_fraction_change",
    "rms_dbfs_change",
    "clip_fraction_change",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260926)
    return parser.parse_args()


def normalize_text(value):
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(value).lower()))


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column, right_item in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def normalized_edit(left, right, character=False):
    left = normalize_text(left)
    right = normalize_text(right)
    if character:
        left_items = list(left.replace(" ", ""))
        right_items = list(right.replace(" ", ""))
    else:
        left_items, right_items = left.split(), right.split()
    return edit_distance(left_items, right_items) / max(
        1, len(left_items), len(right_items)
    )


def embedding(value):
    result = np.asarray(json.loads(value), dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all() or np.linalg.norm(result) == 0:
        raise ValueError("identity_embedding_json must be a finite nonzero vector")
    return result / np.linalg.norm(result)


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


def main():
    args = parse_args()
    frame = pd.read_csv(args.evaluation_csv, dtype={"source_sentence": str})
    required = set(UNIT_FIELDS) | {
        "role", "identity_embedding_json", "hypothesis", "identity_delta",
        "robust_content_error", "silence_fraction", "rms_dbfs", "clip_fraction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing evaluation columns: {sorted(missing)}")
    if set(frame.role) != set(ROLES):
        raise ValueError(f"expected roles {ROLES}, found {sorted(frame.role.unique())}")
    if frame.duplicated(["variant", "unit_id", "role"]).any():
        raise ValueError("duplicate variant/unit/role rows")
    role_counts = frame.groupby(["variant", "unit_id"]).role.nunique()
    if not (role_counts == len(ROLES)).all():
        raise ValueError("every unit must contain all seven roles")
    unit_count = frame[["variant", "unit_id"]].drop_duplicates().shape[0]
    if len(frame) != unit_count * len(ROLES):
        raise ValueError("evaluation grid cardinality mismatch")

    pair_rows = []
    for _, unit in frame.groupby(["variant", "unit_id"], sort=True):
        indexed = unit.set_index("role")
        anchor = indexed.loc["anchor"]
        anchor_embedding = embedding(anchor.identity_embedding_json)
        metadata = {field: anchor[field] for field in UNIT_FIELDS}
        for role in PAIR_ROLES:
            other = indexed.loc[role]
            other_embedding = embedding(other.identity_embedding_json)
            pair_rows.append({
                **metadata,
                "comparison_role": role,
                "wavlm_cosine_distance": 1.0 - float(
                    np.dot(anchor_embedding, other_embedding)
                ),
                "word_edit_distance": normalized_edit(
                    anchor.hypothesis, other.hypothesis
                ),
                "char_edit_distance": normalized_edit(
                    anchor.hypothesis, other.hypothesis, character=True
                ),
                "identity_delta_change": (
                    float(other.identity_delta) - float(anchor.identity_delta)
                ),
                "identity_delta_abs_change": abs(
                    float(other.identity_delta) - float(anchor.identity_delta)
                ),
                "robust_content_error_change": (
                    float(other.robust_content_error)
                    - float(anchor.robust_content_error)
                ),
                "robust_content_error_abs_change": abs(
                    float(other.robust_content_error)
                    - float(anchor.robust_content_error)
                ),
                "silence_fraction_change": (
                    float(other.silence_fraction) - float(anchor.silence_fraction)
                ),
                "rms_dbfs_change": float(other.rms_dbfs) - float(anchor.rms_dbfs),
                "clip_fraction_change": (
                    float(other.clip_fraction) - float(anchor.clip_fraction)
                ),
            })
    pairs = pd.DataFrame(pair_rows)
    if len(pairs) != unit_count * len(PAIR_ROLES):
        raise RuntimeError("incomplete path-length pair table")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "path_length_pairwise_distances.csv", index=False)
    role_summary = pairs.groupby(
        ["variant", "comparison_role"], as_index=False
    )[list(METRICS)].mean()
    role_summary.to_csv(args.output_dir / "path_length_role_summary.csv", index=False)

    pivot = pairs.pivot(
        index=UNIT_FIELDS, columns="comparison_role", values=list(METRICS)
    )
    if pivot.isna().any().any():
        raise ValueError("paired path-length table has missing cells")

    def mean_roles(metric, roles):
        return sum(pivot[(metric, role)] for role in roles) / len(roles)

    two_roles = ("two_hop_via_a", "two_hop_via_b")
    three_roles = ("three_hop_via_a_b", "three_hop_via_b_a")
    contrast_specs = (
        ("three_minus_two", three_roles, two_roles),
        ("two_minus_stochastic", two_roles, ("stochastic",)),
        ("three_minus_stochastic", three_roles, ("stochastic",)),
        ("two_minus_alternate_reference", two_roles, ("alternate_reference",)),
        ("three_minus_alternate_reference", three_roles, ("alternate_reference",)),
    )
    generator = np.random.default_rng(args.seed)
    contrast_rows, cluster_rows = [], []
    for name, left_roles, right_roles in contrast_specs:
        for metric in METRICS:
            differences = (
                mean_roles(metric, left_roles) - mean_roles(metric, right_roles)
            ).rename("difference").reset_index()
            for variant, variant_differences in differences.groupby("variant"):
                clusters = variant_differences.groupby(
                    CLUSTER_FIELDS, as_index=False
                ).difference.mean()
                ci_low, ci_high = bootstrap_ci(
                    clusters.difference, generator, args.bootstrap_repetitions
                )
                p_value = sign_flip_pvalue(
                    clusters.difference, generator, args.bootstrap_repetitions
                )
                contrast_rows.append({
                    "variant": variant,
                    "comparison": name,
                    "metric": metric,
                    "raw_paired_units": len(variant_differences),
                    "source_clusters": len(clusters),
                    "mean_difference": float(clusters.difference.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_sign_flip_mc": p_value,
                    "positive_clusters": int((clusters.difference > 0).sum()),
                })
                clusters.insert(0, "metric", metric)
                clusters.insert(0, "comparison", name)
                clusters.insert(0, "variant", variant)
                cluster_rows.extend(clusters.to_dict("records"))

    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.output_dir / "path_length_contrasts.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(
        args.output_dir / "path_length_source_cluster_differences.csv", index=False
    )
    primary = contrasts[
        (contrasts.comparison == "three_minus_two")
        & contrasts.metric.isin((
            "word_edit_distance", "robust_content_error_change"
        ))
    ]
    primary_supported = (
        len(primary) == 2 and bool((primary.ci_low > 0).all())
    )
    summary = {
        "date": "2026-09-26",
        "status": "complete",
        "design": "paired_one_two_three_hop_path_length_diagnostic",
        "evaluated_outputs": len(frame),
        "variant_units": unit_count,
        "source_clusters_per_variant": {
            variant: int(group[CLUSTER_FIELDS].drop_duplicates().shape[0])
            for variant, group in frame.groupby("variant")
        },
        "roles": list(ROLES),
        "primary_hypothesis_supported": primary_supported,
        "primary_rule": (
            "both three-minus-two co-primary 95% cluster-bootstrap intervals "
            "must exclude zero in the harmful direction"
        ),
        "role_means": role_summary.to_dict("records"),
        "contrasts": contrasts.to_dict("records"),
        "inference_warning": (
            "The 20 source clusters share four speakers and the checkpoint may "
            "have historical sentence exposure; this is not population-level "
            "or perceptual evidence."
        ),
    }
    (args.output_dir / "path_length_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(role_summary.to_string(index=False))
    print(contrasts.to_string(index=False))
    print(f"PATH_LENGTH_ANALYSIS_COMPLETE {args.output_dir}")


if __name__ == "__main__":
    main()
