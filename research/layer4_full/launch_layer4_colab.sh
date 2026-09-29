#!/usr/bin/env bash
set -euo pipefail

project_root="${CYCLEDIFFUSION_ROOT:-/content/CycleDiffusion}"
controlled_root="${CYCLEDIFFUSION_RESULTS_ROOT:-/content/drive/MyDrive/CycleDiffusion_results/layers3_5/full4_controlled}"
layer4_root="${controlled_root}/layer4_component_ablation"

required=(
  "${project_root}/launch_controlled_colab.sh"
  "${project_root}/generate_controlled_grid.py"
  "${project_root}/evaluate_controlled_grid.py"
  "${project_root}/analyze_component_ablation.py"
  "${controlled_root}/formal_evaluation/metrics/formal_eval_raw.csv"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required asset: ${path}" >&2
    exit 3
  fi
done

mkdir -p "${layer4_root}/generation" "${layer4_root}/metrics"

for variant in speaker_only content_only hard_gate; do
  bash "${project_root}/launch_controlled_colab.sh" "${variant}" 37
done

python -u "${project_root}/generate_controlled_grid.py" \
  --project-root "${project_root}" \
  --variant-checkpoint "speaker_only_e51=${controlled_root}/speaker_only_seed37_e51/vc_51.pt" \
  --variant-checkpoint "content_only_e51=${controlled_root}/content_only_seed37_e51/vc_51.pt" \
  --variant-checkpoint "hard_gate_e51=${controlled_root}/hard_gate_seed37_e51/vc_51.pt" \
  --output-dir "${layer4_root}/generation" \
  --diffusion-steps 30 \
  --seed 20260915

python -u "${project_root}/evaluate_controlled_grid.py" \
  --project-root "${project_root}" \
  --generated-dir "${layer4_root}/generation" \
  --output-dir "${layer4_root}/metrics" \
  --batch-size 8

python -u "${project_root}/analyze_component_ablation.py" \
  --formal-csv "${controlled_root}/formal_evaluation/metrics/formal_eval_raw.csv" \
  --component-csv "${layer4_root}/metrics/formal_eval_raw.csv" \
  --controlled-root "${controlled_root}" \
  --output-dir "${layer4_root}/metrics"

echo "LAYER4_PIPELINE_COMPLETE ${layer4_root}"
