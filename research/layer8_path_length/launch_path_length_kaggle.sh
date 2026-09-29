#!/usr/bin/env bash
set -euo pipefail

project_root="${CYCLEDIFFUSION_ROOT:-/kaggle/working/CycleDiffusion}"
overlay_root="${PATH_LENGTH_OVERLAY_ROOT:-/kaggle/working/project_overlay/research/layer8_path_length}"
manifest="${PATH_LENGTH_MANIFEST:-${overlay_root}/layer8_path_length_manifest.csv}"
checkpoint="${PATH_LENGTH_CHECKPOINT:-${project_root}/real_last_cycle_train_dec_4speakers_original/vc_50_0823.pt}"
results_root="${PATH_LENGTH_RESULTS_ROOT:-/kaggle/working/layer8_path_length}"

required=(
  "${overlay_root}/generate_path_length_grid.py"
  "${overlay_root}/evaluate_path_length_grid.py"
  "${overlay_root}/analyze_path_length.py"
  "${manifest}"
  "${checkpoint}"
  "${project_root}/checkpts/vocoder/config.json"
  "${project_root}/checkpts/vocoder/generator"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "MISSING_REQUIRED_ASSET ${path}" >&2
    exit 1
  fi
done

python -m py_compile \
  "${overlay_root}/generate_path_length_grid.py" \
  "${overlay_root}/evaluate_path_length_grid.py" \
  "${overlay_root}/analyze_path_length.py"

python -u "${overlay_root}/generate_path_length_grid.py" \
  --project-root "${project_root}" \
  --unit-manifest "${manifest}" \
  --base-checkpoint "${checkpoint}" \
  --output-dir "${results_root}/generation" \
  --diffusion-steps 30

python -u "${overlay_root}/evaluate_path_length_grid.py" \
  --project-root "${project_root}" \
  --generated-dir "${results_root}/generation" \
  --output-dir "${results_root}/evaluation"

python -u "${overlay_root}/analyze_path_length.py" \
  --evaluation-csv "${results_root}/evaluation/path_length_eval_raw.csv" \
  --output-dir "${results_root}/analysis"

echo "PATH_LENGTH_PIPELINE_COMPLETE ${results_root}"
