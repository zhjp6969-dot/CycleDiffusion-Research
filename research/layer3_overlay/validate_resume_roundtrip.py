"""Run a tiny real-training round trip to validate exact mid-epoch resume."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/content/CycleDiffusion")
CENTROIDS = ROOT / "artifacts/train_centroids.npz"


def run(args: list[str]) -> None:
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout[-5000:])
    if result.returncode:
        raise SystemExit(result.returncode)


def read_steps(output_dir: Path) -> list[int]:
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    return [int(row["global_step"]) for row in rows]


def main() -> None:
    if not CENTROIDS.exists():
        raise FileNotFoundError(CENTROIDS)

    output_dir = ROOT / "runs" / f"resume_smoke_uniform_d794_{int(time.time())}"
    base = [
        sys.executable,
        "-u",
        str(ROOT / "train_reliability_cycle.py"),
        "--data-dir",
        str(ROOT / "VCTK_2F2M"),
        "--encoder-checkpoint",
        str(ROOT / "checkpts/spk_encoder/enc.pt"),
        "--resume",
        str(ROOT / "real_last_cycle_train_dec_4speakers_original/vc_50_0823.pt"),
        "--centroids",
        str(CENTROIDS),
        "--output-dir",
        str(output_dir),
        "--variant",
        "uniform",
        "--lambda-cycle",
        "1",
        "--start-epoch",
        "51",
        "--end-epoch",
        "51",
        "--checkpoint-every",
        "1",
        "--checkpoint-every-steps",
        "1",
        "--batch-size",
        "4",
        "--cycle-batch-size",
        "3",
        "--diffusion-steps",
        "6",
        "--learning-rate",
        "3e-5",
        "--seed",
        "37",
    ]

    run(base + ["--max-steps-per-epoch", "2"])
    state_path = output_dir / "training_state_latest.pt"
    first_state = torch.load(state_path, map_location="cpu", weights_only=True)
    first_steps = read_steps(output_dir)
    assert first_steps == [1, 2], first_steps
    assert int(first_state["completed_step"]) == 2
    assert int(first_state["global_step"]) == 2
    print("FIRST_LEG_PASS", output_dir, first_steps)

    run(
        base
        + [
            "--max-steps-per-epoch",
            "3",
            "--resume-training-state",
            str(state_path),
        ]
    )
    final_state = torch.load(state_path, map_location="cpu", weights_only=True)
    final_steps = read_steps(output_dir)
    assert final_steps == [1, 2, 3], final_steps
    assert int(final_state["completed_step"]) == 3
    assert int(final_state["global_step"]) == 3
    print("RESUME_ROUNDTRIP_PASS", output_dir, final_steps)


if __name__ == "__main__":
    main()
