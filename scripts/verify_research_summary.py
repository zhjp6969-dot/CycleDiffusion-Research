#!/usr/bin/env python3
"""Check published summary consistency; does not recompute raw-audio statistics."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"


def rows(relative):
    with (DATA / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(relative, **keys):
    matches = [r for r in rows(relative) if all(r[k] == v for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"{relative}: expected one match for {keys}, got {len(matches)}")
    return matches[0]


def effect(row):
    prefix = "hierarchical_" if "hierarchical_ci_low" in row else ""
    values = tuple(float(row[k]) for k in
                   ("mean_difference", prefix + "ci_low", prefix + "ci_high"))
    if not all(math.isfinite(v) for v in values) or not values[1] <= values[0] <= values[2]:
        raise ValueError(f"Invalid effect or interval: {values}")
    return values


def load_effects():
    specs = [
        ("Discovery: joint", "layer6_formal_eval_paired_comparisons.csv", "joint_minus_uniform"),
        ("Discovery: content-only", "layer4_full_primary_comparisons.csv", "content_only_e51_minus_uniform"),
        ("Confirmation: content-only", "layer10_confirmation/analysis/layer10_pooled_effects.csv", "content_only_minus_uniform"),
    ]
    return [(label, {metric: effect(select(path, comparison=contrast, metric=metric))
                     for metric in ("identity_delta", "robust_content_error")})
            for label, path, contrast in specs]


def load_path_effects():
    return [(metric, effect(select(
        "layer8_path_length/analysis/path_length_contrasts.csv",
        comparison="three_minus_two", metric=metric)))
        for metric in ("word_edit_distance", "robust_content_error_change")]


def check_close(actual, expected):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-10):
        raise ValueError(f"Inconsistent summaries: {actual} != {expected}")


def verify():
    effects = load_effects()
    formal = json.loads((DATA / "layer6_formal_summary.json").read_text())
    if formal["evaluation_outputs"] != 540 or formal["independent_source_clusters"] != 20:
        raise ValueError("Unexpected discovery counts")
    for metric, values in effects[0][1].items():
        primary = formal["primary_result"]
        expected = [primary[metric + "_mean_difference"],
                    *primary[metric + "_cluster_bootstrap_95ci"]]
        for actual, target in zip(values, expected):
            check_close(actual, target)
    confirmation = json.loads((DATA / "layer10_confirmation/analysis/layer10_summary.json").read_text())
    if confirmation["evaluation_outputs"] != 1080 or confirmation["analysis_units"] != 60:
        raise ValueError("Unexpected confirmation counts")
    if confirmation["identity_promotion_criterion_met"]:
        raise ValueError("Confirmation conclusion changed")
    units = rows("layer10_confirmation/analysis/layer10_source_unit_differences.csv")
    for metric, values in effects[2][1].items():
        summary = next(r for r in confirmation["pooled_effects"] if r["metric"] == metric)
        for actual, target in zip(values, effect(summary)):
            check_close(actual, target)
        subset = [r for r in units if r["metric"] == metric]
        keys = {(r["seed"], r["source_speaker"], r["source_sentence"]) for r in subset}
        if len(subset) != 60 or len(keys) != 60:
            raise ValueError("Missing or duplicate confirmation units")
        check_close(sum(float(r["difference"]) for r in subset) / 60, values[0])
    for metric, values in load_path_effects():
        row = select("layer8_path_length/analysis/path_length_contrasts.csv",
                     comparison="three_minus_two", metric=metric)
        if int(row["raw_paired_units"]) != 60 or int(row["source_clusters"]) != 20:
            raise ValueError("Unexpected path-length counts")
        if values[1] <= 0:
            raise ValueError("Path-length conclusion changed")
        clusters = [r for r in rows(
            "layer8_path_length/analysis/path_length_source_cluster_differences.csv")
            if r["metric"] == metric and r["comparison"] == "three_minus_two"]
        keys = {(r["src"], r["source_sentence"]) for r in clusters}
        if len(clusters) != 20 or len(keys) != 20:
            raise ValueError("Missing or duplicate path-length clusters")
        check_close(sum(float(r["difference"]) for r in clusters) / 20, values[0])
    for label, metrics in effects:
        for metric, (mean, low, high) in metrics.items():
            print(f"{label:28s} {metric:24s} {mean:+.5f} [{low:+.5f}, {high:+.5f}]")
    for metric, (mean, low, high) in load_path_effects():
        print(f"Three minus two hops: {metric}: {mean:+.5f} [{low:+.5f}, {high:+.5f}]")
    print("PASS: summary consistency, confirmation unit means, and path-length cluster means.")
    print("This check does not rerun raw evaluation or recompute bootstrap intervals.")


if __name__ == "__main__":
    verify()
