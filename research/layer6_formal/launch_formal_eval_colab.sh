#!/usr/bin/env bash
set -euo pipefail

project_root=/content/CycleDiffusion
controlled_root=/content/drive/MyDrive/CycleDiffusion_results/layers3_5/full4_controlled
evaluation_root="${controlled_root}/formal_evaluation"
uniform_run="${controlled_root}/uniform_seed37_e51"
joint_run="${controlled_root}/joint_seed37_e51"

required=(
  "${project_root}/generate_controlled_grid.py"
  "${project_root}/evaluate_controlled_grid.py"
  "${project_root}/analyze_controlled_grid.py"
  "${project_root}/real_last_cycle_train_dec_4speakers_original/vc_50_0823.pt"
  "${uniform_run}/vc_51.pt"
  "${joint_run}/vc_51.pt"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required asset: ${path}" >&2
    exit 3
  fi
done

mkdir -p "${evaluation_root}/generation" "${evaluation_root}/metrics"

python -u "${project_root}/generate_controlled_grid.py" \
  --project-root "${project_root}" \
  --base-checkpoint "${project_root}/real_last_cycle_train_dec_4speakers_original/vc_50_0823.pt" \
  --uniform-checkpoint "${uniform_run}/vc_51.pt" \
  --joint-checkpoint "${joint_run}/vc_51.pt" \
  --output-dir "${evaluation_root}/generation" \
  --diffusion-steps 30 \
  --seed 20260915

python -u "${project_root}/evaluate_controlled_grid.py" \
  --project-root "${project_root}" \
  --generated-dir "${evaluation_root}/generation" \
  --output-dir "${evaluation_root}/metrics" \
  --batch-size 8

python -u "${project_root}/analyze_controlled_grid.py" \
  --evaluation-csv "${evaluation_root}/metrics/formal_eval_raw.csv" \
  --uniform-run-dir "${uniform_run}" \
  --joint-run-dir "${joint_run}" \
  --output-dir "${evaluation_root}/metrics"

echo "FORMAL_PIPELINE_COMPLETE ${evaluation_root}"
