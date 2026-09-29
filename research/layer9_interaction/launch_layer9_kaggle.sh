#!/usr/bin/env bash
set -euo pipefail

project_root="${CYCLEDIFFUSION_ROOT:-/kaggle/working/CycleDiffusion}"
controlled_root="${CYCLEDIFFUSION_RESULTS_ROOT:-/kaggle/working/full4_controlled}"
layer9_root="${controlled_root}/layer9_interaction"

required=(
  "${project_root}/train_reliability_cycle.py"
  "${project_root}/generate_controlled_grid.py"
  "${project_root}/evaluate_controlled_grid.py"
  "${project_root}/analyze_interaction.py"
  "${controlled_root}/formal_evaluation/metrics/formal_eval_raw.csv"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required asset: ${path}" >&2
    exit 3
  fi
done

mkdir -p "${layer9_root}/generation" "${layer9_root}/metrics"

run_cell() {
  local label="$1"
  local variant="$2"
  local lambda_cycle="$3"
  local output_dir="${controlled_root}/${label}_seed37_e51"
  local state_path="${output_dir}/training_state_latest.pt"
  local final_checkpoint="${output_dir}/vc_51.pt"
  mkdir -p "${output_dir}"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "already complete: ${final_checkpoint}"
    return
  fi
  local resume_args=()
  if [[ -f "${state_path}" ]]; then
    resume_args=(--resume-training-state "${state_path}")
    echo "resuming from ${state_path}"
  else
    echo "starting ${label}: variant=${variant}, lambda_cycle=${lambda_cycle}"
  fi
  python -u "${project_root}/train_reliability_cycle.py" \
    --data-dir "${project_root}/VCTK_2F2M" \
    --encoder-checkpoint "${project_root}/checkpts/spk_encoder/enc.pt" \
    --resume "${project_root}/real_last_cycle_train_dec_4speakers_original/vc_50_0823.pt" \
    --centroids "${project_root}/artifacts/train_centroids.npz" \
    --output-dir "${output_dir}" \
    --variant "${variant}" \
    --lambda-cycle "${lambda_cycle}" \
    --start-epoch 51 \
    --end-epoch 51 \
    --checkpoint-every 1 \
    --checkpoint-every-steps 25 \
    --batch-size 4 \
    --cycle-batch-size 3 \
    --diffusion-steps 6 \
    --learning-rate 3e-5 \
    --seed 37 \
    "${resume_args[@]}"
}

run_cell content_only_lambda100 content_only 1.0
run_cell uniform_lambda025 uniform 0.25
run_cell content_only_lambda025 content_only 0.25

python -u "${project_root}/generate_controlled_grid.py" \
  --project-root "${project_root}" \
  --variant-checkpoint "content_only_lambda100_e51=${controlled_root}/content_only_lambda100_seed37_e51/vc_51.pt" \
  --variant-checkpoint "uniform_lambda025_e51=${controlled_root}/uniform_lambda025_seed37_e51/vc_51.pt" \
  --variant-checkpoint "content_only_lambda025_e51=${controlled_root}/content_only_lambda025_seed37_e51/vc_51.pt" \
  --output-dir "${layer9_root}/generation" \
  --diffusion-steps 30 \
  --seed 20260915

python -u "${project_root}/evaluate_controlled_grid.py" \
  --project-root "${project_root}" \
  --generated-dir "${layer9_root}/generation" \
  --output-dir "${layer9_root}/metrics" \
  --batch-size 8

python -u "${project_root}/analyze_interaction.py" \
  --formal-csv "${controlled_root}/formal_evaluation/metrics/formal_eval_raw.csv" \
  --interaction-csv "${layer9_root}/metrics/formal_eval_raw.csv" \
  --controlled-root "${controlled_root}" \
  --output-dir "${layer9_root}/metrics"

tar --exclude='training_state_latest.pt' \
  -C "${controlled_root}" \
  -czf /kaggle/working/layer9_interaction_outputs.tgz \
  layer9_interaction \
  content_only_lambda100_seed37_e51 \
  uniform_lambda025_seed37_e51 \
  content_only_lambda025_seed37_e51
sha256sum /kaggle/working/layer9_interaction_outputs.tgz \
  > /kaggle/working/layer9_interaction_outputs.sha256

python - <<'PY'
import json
from pathlib import Path
Path('/kaggle/working/layer9_status.json').write_text(json.dumps({
    'status': 'complete',
    'archive': '/kaggle/working/layer9_interaction_outputs.tgz',
    'summary': '/kaggle/working/full4_controlled/layer9_interaction/metrics/layer9_summary.json',
}, indent=2) + '\n')
PY

echo "LAYER9_KAGGLE_PIPELINE_COMPLETE ${layer9_root}"
