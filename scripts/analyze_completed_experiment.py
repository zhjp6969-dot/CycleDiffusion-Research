#!/usr/bin/env python3
"""Audit and summarize the completed CycleDiffusion alpha experiment.

This script intentionally uses only the Python standard library. It validates the
three row-level tables, applies the pre-declared clipping exclusion rule used in
the original analysis, performs target-reference sensitivity and Pareto analyses,
and writes compact CSV/JSON/SVG artifacts for the repository.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PUBLIC = ROOT / "data" / "public"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SPEAKERS = ("p236", "p239", "p259", "p263")
KEY_FIELDS = ("src", "tgt", "source_idx", "ref_idx", "alpha")
CLIP_THRESHOLD = 0.01


def read_csv(name: str) -> list[dict]:
    with (RAW / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_input(public_name: str, raw_name: str) -> list[dict]:
    path = PUBLIC / public_name
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            return rows
    return read_csv(raw_name)


def converted_key(row: dict) -> tuple:
    return (
        row["src"],
        row["tgt"],
        int(row["source_idx"]),
        int(row["ref_idx"]),
        float(row["alpha"]),
    )


def paired_key(row: dict) -> tuple:
    return (
        row["src"],
        row["tgt"],
        int(row["source_idx"]),
        int(row["ref_idx"]),
    )


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("mean of empty values")
    return statistics.fmean(values)


def bootstrap_ci(values, *, seed: int, n: int = 20_000) -> tuple[float, float]:
    values = list(values)
    rng = random.Random(seed)
    draws = sorted(mean(values[rng.randrange(len(values))] for _ in values) for _ in range(n))
    return draws[int(0.025 * n)], draws[int(0.975 * n)]


def fmt_alpha(value: float) -> str:
    return f"{value:g}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_tables(identity: list[dict], content: list[dict], acoustic: list[dict]) -> dict:
    converted_content = [row for row in content if float(row["alpha"]) >= 0]
    converted_acoustic = [row for row in acoustic if float(row["alpha"]) >= 0]
    key_sets = [
        {converted_key(row) for row in identity},
        {converted_key(row) for row in converted_content},
        {converted_key(row) for row in converted_acoustic},
    ]
    duplicate_counts = [
        len(identity) - len(key_sets[0]),
        len(converted_content) - len(key_sets[1]),
        len(converted_acoustic) - len(key_sets[2]),
    ]
    if duplicate_counts != [0, 0, 0]:
        raise ValueError(f"duplicate converted keys: {duplicate_counts}")
    if not (key_sets[0] == key_sets[1] == key_sets[2]):
        raise ValueError("identity, content, and acoustic converted keys do not align")

    expected = {
        (src, tgt, source_idx, ref_idx, alpha)
        for src in SPEAKERS
        for tgt in SPEAKERS
        if src != tgt
        for source_idx in range(1, 6)
        for ref_idx in range(1, 4)
        for alpha in ALPHAS
    }
    if key_sets[0] != expected:
        missing = len(expected - key_sets[0])
        extra = len(key_sets[0] - expected)
        raise ValueError(f"unexpected experiment grid: {missing=} {extra=}")

    return {
        "converted_outputs": len(key_sets[0]),
        "original_source_rows": sum(float(row["alpha"]) < 0 for row in content),
        "speaker_count": len(SPEAKERS),
        "direction_count": len(SPEAKERS) * (len(SPEAKERS) - 1),
        "independent_source_clusters": len({(row["src"], int(row["source_idx"])) for row in identity}),
        "direction_source_clusters": len({(row["src"], row["tgt"], int(row["source_idx"])) for row in identity}),
        "paired_direction_source_reference_units": len({paired_key(row) for row in identity}),
        "alpha_values": list(ALPHAS),
        "tables_key_aligned": True,
        "duplicate_converted_keys": 0,
    }


def detect_acoustic_anomalies(acoustic: list[dict]) -> tuple[set[tuple], list[dict]]:
    converted = [row for row in acoustic if float(row["alpha"]) >= 0]
    severe = [row for row in converted if float(row["clip_fraction"]) > CLIP_THRESHOLD]
    bad_keys = {paired_key(row) for row in severe}
    minor = [
        row
        for row in converted
        if 0 < float(row["clip_fraction"]) <= CLIP_THRESHOLD
    ]
    return bad_keys, minor


def reference_sensitivity(identity: list[dict], bad_keys: set[tuple]) -> tuple[list[dict], list[dict]]:
    kept = [row for row in identity if paired_key(row) not in bad_keys]
    curves = []
    best_rows = []
    for src in SPEAKERS:
        for tgt in SPEAKERS:
            if src == tgt:
                continue
            direction_means = {}
            reference_means = defaultdict(dict)
            for alpha in ALPHAS:
                direction_means[alpha] = mean(
                    float(row["delta"])
                    for row in kept
                    if row["src"] == src and row["tgt"] == tgt and float(row["alpha"]) == alpha
                )
                for ref_idx in range(1, 4):
                    selected = [
                        float(row["delta"])
                        for row in kept
                        if row["src"] == src
                        and row["tgt"] == tgt
                        and int(row["ref_idx"]) == ref_idx
                        and float(row["alpha"]) == alpha
                    ]
                    reference_means[ref_idx][alpha] = mean(selected)
                    curves.append(
                        {
                            "src": src,
                            "tgt": tgt,
                            "ref_idx": ref_idx,
                            "alpha": fmt_alpha(alpha),
                            "n_sources": len(selected),
                            "mean_identity_delta": f"{reference_means[ref_idx][alpha]:.9f}",
                            "gain_vs_alpha0": f"{reference_means[ref_idx][alpha] - reference_means[ref_idx][0.0]:.9f}",
                        }
                    )
            direction_best = max(ALPHAS, key=lambda a: (direction_means[a], -a))
            ref_best = {
                ref_idx: max(ALPHAS, key=lambda a: (reference_means[ref_idx][a], -a))
                for ref_idx in range(1, 4)
            }
            ref_gains = {
                ref_idx: reference_means[ref_idx][direction_best] - reference_means[ref_idx][0.0]
                for ref_idx in range(1, 4)
            }
            best_rows.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "direction_best_alpha": fmt_alpha(direction_best),
                    "ref1_best_alpha": fmt_alpha(ref_best[1]),
                    "ref2_best_alpha": fmt_alpha(ref_best[2]),
                    "ref3_best_alpha": fmt_alpha(ref_best[3]),
                    "all_references_agree": len(set(ref_best.values())) == 1,
                    "ref1_gain_at_direction_best": f"{ref_gains[1]:.9f}",
                    "ref2_gain_at_direction_best": f"{ref_gains[2]:.9f}",
                    "ref3_gain_at_direction_best": f"{ref_gains[3]:.9f}",
                    "all_reference_gains_positive": direction_best != 0 and all(value > 0 for value in ref_gains.values()),
                }
            )
    return curves, best_rows


def pareto_analysis(identity: list[dict], content: list[dict], bad_keys: set[tuple]) -> list[dict]:
    kept_identity = [row for row in identity if paired_key(row) not in bad_keys]
    kept_content = [
        row
        for row in content
        if float(row["alpha"]) >= 0 and paired_key(row) not in bad_keys
    ]
    rows = []
    for src in SPEAKERS:
        for tgt in SPEAKERS:
            if src == tgt:
                continue
            points = []
            for alpha in ALPHAS:
                identity_values = [
                    float(row["delta"])
                    for row in kept_identity
                    if row["src"] == src and row["tgt"] == tgt and float(row["alpha"]) == alpha
                ]
                content_values = [
                    max(min(float(row["wer"]), 1.0), min(float(row["cer"]), 1.0))
                    for row in kept_content
                    if row["src"] == src and row["tgt"] == tgt and float(row["alpha"]) == alpha
                ]
                raw_wer = [
                    float(row["wer"])
                    for row in kept_content
                    if row["src"] == src and row["tgt"] == tgt and float(row["alpha"]) == alpha
                ]
                points.append((alpha, mean(identity_values), mean(content_values), mean(raw_wer), len(identity_values)))
            for alpha, identity_mean, robust_error, mean_wer, n in points:
                dominated = any(
                    other_identity >= identity_mean
                    and other_error <= robust_error
                    and (other_identity > identity_mean or other_error < robust_error)
                    for other_alpha, other_identity, other_error, _, _ in points
                    if other_alpha != alpha
                )
                rows.append(
                    {
                        "src": src,
                        "tgt": tgt,
                        "alpha": fmt_alpha(alpha),
                        "n_outputs": n,
                        "mean_identity_delta": f"{identity_mean:.9f}",
                        "mean_robust_content_error": f"{robust_error:.9f}",
                        "mean_raw_wer": f"{mean_wer:.9f}",
                        "pareto_optimal": not dominated,
                    }
                )
    return rows


def loso_recheck(identity: list[dict], content: list[dict], bad_keys: set[tuple]) -> dict:
    kept_identity = [row for row in identity if paired_key(row) not in bad_keys]
    kept_content = [
        row for row in content if float(row["alpha"]) >= 0 and paired_key(row) not in bad_keys
    ]
    identity_by = defaultdict(list)
    content_by = defaultdict(list)
    for row in kept_identity:
        identity_by[(row["src"], row["tgt"], int(row["source_idx"]), float(row["alpha"]))].append(float(row["delta"]))
    for row in kept_content:
        robust_error = max(min(float(row["wer"]), 1.0), min(float(row["cer"]), 1.0))
        content_by[(row["src"], row["tgt"], int(row["source_idx"]), float(row["alpha"]))].append(robust_error)

    folds = []
    for src in SPEAKERS:
        for tgt in SPEAKERS:
            if src == tgt:
                continue
            id_curve = {
                alpha: {idx: mean(identity_by[(src, tgt, idx, alpha)]) for idx in range(1, 6)}
                for alpha in ALPHAS
            }
            content_curve = {
                alpha: {idx: mean(content_by[(src, tgt, idx, alpha)]) for idx in range(1, 6)}
                for alpha in ALPHAS
            }
            for held_out in range(1, 6):
                train = [idx for idx in range(1, 6) if idx != held_out]
                selected = max(ALPHAS, key=lambda alpha: (mean(id_curve[alpha][idx] for idx in train), -alpha))
                folds.append(
                    {
                        "src": src,
                        "tgt": tgt,
                        "source_idx": held_out,
                        "selected_alpha": fmt_alpha(selected),
                        "identity_gain": id_curve[selected][held_out] - id_curve[0.0][held_out],
                        "content_change": content_curve[selected][held_out] - content_curve[0.0][held_out],
                    }
                )
    clustered = []
    for src in SPEAKERS:
        for source_idx in range(1, 6):
            subset = [row for row in folds if row["src"] == src and row["source_idx"] == source_idx]
            clustered.append(
                {
                    "src": src,
                    "source_idx": source_idx,
                    "identity_gain": mean(row["identity_gain"] for row in subset),
                    "content_change": mean(row["content_change"] for row in subset),
                }
            )
    identity_gains = [row["identity_gain"] for row in clustered]
    content_changes = [row["content_change"] for row in clustered]
    identity_ci = bootstrap_ci(identity_gains, seed=20260907)
    content_ci = bootstrap_ci(content_changes, seed=20261007)
    return {
        "n_source_clusters": len(clustered),
        "mean_identity_gain": mean(identity_gains),
        "identity_gain_bootstrap_95ci": identity_ci,
        "positive_identity_clusters": sum(value > 0 for value in identity_gains),
        "mean_robust_content_change": mean(content_changes),
        "content_change_bootstrap_95ci": content_ci,
        "positive_content_error_clusters": sum(value > 0 for value in content_changes),
    }


def svg_heatmap(best_rows: list[dict]) -> None:
    width, height = 760, 510
    x0, y0, row_h, cell_w = 150, 62, 34, 120
    colors = {0.0: "#440154", 0.25: "#3b528b", 0.5: "#21918c", 0.75: "#5ec962", 1.0: "#fde725"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a}.title{font-size:20px;font-weight:700}.h{font-size:13px;font-weight:700}.t{font-size:12px}</style>',
        '<text x="20" y="28" class="title">Target-reference sensitivity after paired anomaly exclusion</text>',
    ]
    headers = ["Reference 1", "Reference 2", "Reference 3", "All references"]
    for col, label in enumerate(headers):
        parts.append(f'<text x="{x0 + col * cell_w + cell_w / 2}" y="52" text-anchor="middle" class="h">{label}</text>')
    for row_i, row in enumerate(best_rows):
        y = y0 + row_i * row_h
        parts.append(f'<text x="138" y="{y + 22}" text-anchor="end" class="h">{row["src"]} → {row["tgt"]}</text>')
        values = [row["ref1_best_alpha"], row["ref2_best_alpha"], row["ref3_best_alpha"], row["direction_best_alpha"]]
        for col, value in enumerate(values):
            alpha = float(value)
            x = x0 + col * cell_w
            parts.append(f'<rect x="{x + 3}" y="{y + 2}" width="{cell_w - 6}" height="28" rx="4" fill="{colors[alpha]}"/>')
            text_color = "#111111" if alpha >= 0.75 else "#ffffff"
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 21}" text-anchor="middle" class="h" fill="{text_color}" style="fill:{text_color}">α={value}</text>')
    parts.append('<text x="20" y="486" class="t">Cell value: alpha with the highest mean WavLM target-minus-source cosine delta. Descriptive; five source utterances per direction.</text>')
    parts.append("</svg>")
    (FIG / "reference_alpha_heatmap.svg").write_text("\n".join(parts), encoding="utf-8")


def svg_pareto(pareto_rows: list[dict]) -> None:
    width, height = 1000, 730
    panel_w, panel_h = 235, 200
    left, top = 42, 66
    colors = {"0": "#440154", "0.25": "#3b528b", "0.5": "#21918c", "0.75": "#5ec962", "1": "#d4b800"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a}.title{font-size:20px;font-weight:700}.h{font-size:13px;font-weight:700}.t{font-size:10px}.axis{stroke:#aab7b8;stroke-width:1}</style>',
        '<text x="22" y="30" class="title">Identity–content Pareto analysis by conversion direction</text>',
        '<text x="22" y="49" class="t">Higher is better for identity (vertical); lower is better for robust content error (horizontal). Ringed points are Pareto-optimal.</text>',
    ]
    directions = [(s, t) for s in SPEAKERS for t in SPEAKERS if s != t]
    for index, (src, tgt) in enumerate(directions):
        col, row_i = index % 4, index // 4
        px, py = left + col * panel_w, top + row_i * panel_h
        subset = [r for r in pareto_rows if r["src"] == src and r["tgt"] == tgt]
        xs = [float(r["mean_robust_content_error"]) for r in subset]
        ys = [float(r["mean_identity_delta"]) for r in subset]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xpad = max((xmax - xmin) * 0.15, 0.005)
        ypad = max((ymax - ymin) * 0.15, 0.003)
        xmin, xmax = xmin - xpad, xmax + xpad
        ymin, ymax = ymin - ypad, ymax + ypad
        parts.extend([
            f'<text x="{px + 5}" y="{py + 14}" class="h">{src} → {tgt}</text>',
            f'<line x1="{px + 28}" y1="{py + 166}" x2="{px + 218}" y2="{py + 166}" class="axis"/>',
            f'<line x1="{px + 28}" y1="{py + 30}" x2="{px + 28}" y2="{py + 166}" class="axis"/>',
        ])
        for r in subset:
            xval = float(r["mean_robust_content_error"])
            yval = float(r["mean_identity_delta"])
            cx = px + 28 + (xval - xmin) / (xmax - xmin) * 190
            cy = py + 166 - (yval - ymin) / (ymax - ymin) * 136
            a = r["alpha"]
            ring = "#111111" if r["pareto_optimal"] else "#ffffff"
            stroke_w = 2.5 if r["pareto_optimal"] else 1
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{colors[a]}" stroke="{ring}" stroke-width="{stroke_w}"/>')
            parts.append(f'<text x="{cx + 8:.1f}" y="{cy - 6:.1f}" class="t">{a}</text>')
        parts.append(f'<text x="{px + 123}" y="{py + 184}" text-anchor="middle" class="t">content error →</text>')
    parts.append('<text x="25" y="700" class="t">Content metric = mean max(clipped WER, clipped CER), each clipped to [0,1]. Two direction/source/reference units with &gt;1% clipping are excluded at every alpha.</text>')
    parts.append("</svg>")
    (FIG / "identity_content_pareto.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    identity = read_input("identity_scores.csv", "all12_disjoint_samples.csv")
    content = read_input("content_scores.csv", "all12_whisper_content_raw.csv")
    acoustic = read_input("acoustic_scores.csv", "all12_audio_sanity_raw.csv")

    # Publish only numeric scores and experiment keys. VCTK transcripts, audio
    # paths, and Whisper hypotheses remain in the private source table.
    write_csv(PUBLIC / "identity_scores.csv", identity, list(identity[0]))
    content_fields = [
        "src", "tgt", "source_idx", "ref_idx", "alpha",
        "word_edits", "wer", "char_edits", "cer",
    ]
    acoustic_fields = [
        "src", "tgt", "source_idx", "ref_idx", "alpha", "duration_s",
        "rms_dbfs", "peak_dbfs", "clip_fraction", "silence_fraction",
        "dc_offset", "source_duration_s", "duration_ratio",
    ]
    write_csv(PUBLIC / "content_scores.csv", content, content_fields)
    write_csv(PUBLIC / "acoustic_scores.csv", acoustic, acoustic_fields)

    audit = validate_tables(identity, content, acoustic)
    bad_keys, minor_clipping = detect_acoustic_anomalies(acoustic)
    reproduction_dir = ROOT / "scripts" / "reproduction"
    recovered_generation_files = (
        reproduction_dir / "inference_ablation.py",
        reproduction_dir / "make_all_centroids.py",
        reproduction_dir / "make_all_blend_embeddings.py",
        reproduction_dir / "run_all12.sh",
        reproduction_dir / "run_alpha_all12_remaining.sh",
    )
    audit.update(
        {
            "clipping_rule": f"exclude a direction/source/reference unit at every alpha if any output has clip_fraction > {CLIP_THRESHOLD}",
            "excluded_paired_keys": [list(key) for key in sorted(bad_keys)],
            "excluded_outputs": len(bad_keys) * len(ALPHAS),
            "minor_clipping_outputs_not_excluded": len(minor_clipping),
            "generation_code_present": all(path.is_file() for path in recovered_generation_files),
            "generation_protocol_inspectable": True,
            "generation_weights_present": False,
            "environment_lockfile_present": False,
            "human_listener_evaluation_performed": False,
        }
    )

    curves, reference_best = reference_sensitivity(identity, bad_keys)
    pareto = pareto_analysis(identity, content, bad_keys)
    loso = loso_recheck(identity, content, bad_keys)

    write_csv(
        OUT / "reference_sensitivity.csv",
        curves,
        ["src", "tgt", "ref_idx", "alpha", "n_sources", "mean_identity_delta", "gain_vs_alpha0"],
    )
    write_csv(
        OUT / "reference_best_alpha.csv",
        reference_best,
        list(reference_best[0]),
    )
    write_csv(
        OUT / "identity_content_pareto.csv",
        pareto,
        list(pareto[0]),
    )

    reference_agreement = sum(row["all_references_agree"] for row in reference_best)
    nonzero = [row for row in reference_best if float(row["direction_best_alpha"]) != 0]
    universal_positive = sum(row["all_reference_gains_positive"] for row in nonzero)
    pareto_by_direction = {
        f"{src}->{tgt}": [float(row["alpha"]) for row in pareto if row["src"] == src and row["tgt"] == tgt and row["pareto_optimal"]]
        for src in SPEAKERS
        for tgt in SPEAKERS
        if src != tgt
    }
    summary = {
        "audit": audit,
        "reference_sensitivity": {
            "directions_with_complete_reference_alpha_agreement": reference_agreement,
            "total_directions": len(reference_best),
            "nonzero_direction_optima": len(nonzero),
            "nonzero_directions_positive_for_all_references": universal_positive,
        },
        "identity_only_loso_after_exclusion": loso,
        "pareto_alphas_by_direction": pareto_by_direction,
        "interpretation": [
            "A single fixed alpha is not supported as a universal improvement.",
            "Reference choice changes the descriptive alpha optimum in most directions.",
            "Identity-only calibration yields a small average identity gain with a measurable content-error increase.",
            "Results are diagnostic and objective-metric based; no perceptual superiority claim is supported.",
        ],
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    svg_heatmap(reference_best)
    svg_pareto(pareto)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
