#!/usr/bin/env bash
set -euo pipefail

project_root="${CYCLEDIFFUSION_ROOT:-/kaggle/input/datasets/jinpeng12353/cyclediffusion-private-research-assets/CycleDiffusion_kaggle_assets/CycleDiffusion}"
overlay_root="${LAYER10_OVERLAY_ROOT:-/kaggle/input/datasets/jinpeng12353/cyclediffusion-layer-10-evaluation-overlay}"
data_root="${LAYER10_DATA_ROOT:-/kaggle/input/datasets/jinpeng12353/cyclediffusion-layer-10-preprocessed-cohort}"
training_root="${LAYER10_TRAINING_ROOT:-/kaggle/input/datasets/jinpeng12353/cyclediffusion-layer-10-training-state}"
branch_root="${LAYER10_BRANCH_ROOT:-/kaggle/input/datasets/jinpeng12353/cyclediffusion-layer-10-branch-state}"
manifest_csv="${LAYER10_MANIFEST_CSV:-${overlay_root}/layer10_confirmation_manifest.csv}"
results_root="${LAYER10_EVAL_ROOT:-/kaggle/working/layer10_confirmation}"

required=(
  "${overlay_root}/generate_confirmation_grid.py"
  "${overlay_root}/evaluate_confirmation_grid.py"
  "${overlay_root}/analyze_confirmation.py"
  "${manifest_csv}"
  "${data_root}/preprocess_summary.json"
  "${training_root}/seed17/uniform_lambda100/vc_step461.pt"
  "${training_root}/seed17/content_only_lambda100/vc_step461.pt"
  "${branch_root}/seed37/uniform_lambda100/vc_step461.pt"
  "${branch_root}/seed37/content_only_lambda100/vc_step461.pt"
  "${branch_root}/seed73/uniform_lambda100/vc_step461.pt"
  "${branch_root}/seed73/content_only_lambda100/vc_step461.pt"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required Layer 10 evaluation asset: ${path}" >&2
    exit 3
  fi
done

export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
mkdir -p "${results_root}/generation" "${results_root}/evaluation" "${results_root}/analysis"

python -u "${overlay_root}/generate_confirmation_grid.py" \
  --project-root "${project_root}" \
  --data-root "${data_root}" \
  --manifest-csv "${manifest_csv}" \
  --training-state-root "${training_root}" \
  --branch-state-root "${branch_root}" \
  --output-dir "${results_root}/generation"

python -u "${overlay_root}/evaluate_confirmation_grid.py" \
  --data-root "${data_root}" \
  --generated-dir "${results_root}/generation" \
  --output-dir "${results_root}/evaluation"

python -u "${overlay_root}/analyze_confirmation.py" \
  --evaluation-csv "${results_root}/evaluation/formal_eval_raw.csv" \
  --output-dir "${results_root}/analysis"

echo "LAYER10_CONFIRMATION_COMPLETE ${results_root}"
