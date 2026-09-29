#!/usr/bin/env bash
set -euo pipefail

project_root="${CYCLEDIFFUSION_ROOT:-/kaggle/working/CycleDiffusion}"
controlled_root="${CYCLEDIFFUSION_RESULTS_ROOT:-/kaggle/working/full4_controlled}"
layer5_root="${controlled_root}/layer5_lambda_ablation"

required=(
  "${project_root}/train_reliability_cycle.py"
  "${project_root}/generate_controlled_grid.py"
  "${project_root}/evaluate_controlled_grid.py"
  "${project_root}/analyze_lambda_ablation.py"
  "${controlled_root}/formal_evaluation/metrics/formal_eval_raw.csv"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required asset: ${path}" >&2
    exit 3
  fi
done

mkdir -p "${layer5_root}/generation" "${layer5_root}/metrics"

run_lambda() {
  local label="$1"
  local lambda_cycle="$2"
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
    echo "starting ${label} with lambda_cycle=${lambda_cycle}"
  fi
  python -u "${project_root}/train_reliability_cycle.py" \
    --data-dir "${project_root}/VCTK_2F2M" \
    --encoder-checkpoint "${project_root}/checkpts/spk_encoder/enc.pt" \
    --resume "${project_root}/real_last_cycle_train_dec_4speakers_original/vc_50_0823.pt" \
    --centroids "${project_root}/artifacts/train_centroids.npz" \
    --output-dir "${output_dir}" \
    --variant joint \
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

run_lambda joint_lambda050 0.5
run_lambda joint_lambda025 0.25
run_lambda joint_lambda000 0

python -u "${project_root}/generate_controlled_grid.py" \
  --project-root "${project_root}" \
  --variant-checkpoint "joint_lambda050_e51=${controlled_root}/joint_lambda050_seed37_e51/vc_51.pt" \
  --variant-checkpoint "joint_lambda025_e51=${controlled_root}/joint_lambda025_seed37_e51/vc_51.pt" \
  --variant-checkpoint "joint_lambda000_e51=${controlled_root}/joint_lambda000_seed37_e51/vc_51.pt" \
  --output-dir "${layer5_root}/generation" \
  --diffusion-steps 30 \
  --seed 20260915

python -u "${project_root}/evaluate_controlled_grid.py" \
  --project-root "${project_root}" \
  --generated-dir "${layer5_root}/generation" \
  --output-dir "${layer5_root}/metrics" \
  --batch-size 8

python -u "${project_root}/analyze_lambda_ablation.py" \
  --formal-csv "${controlled_root}/formal_evaluation/metrics/formal_eval_raw.csv" \
  --lambda-csv "${layer5_root}/metrics/formal_eval_raw.csv" \
  --controlled-root "${controlled_root}" \
  --output-dir "${layer5_root}/metrics"

tar --exclude='training_state_latest.pt' \
  -C "${controlled_root}" \
  -czf /kaggle/working/layer5_lambda_ablation_outputs.tgz \
  layer5_lambda_ablation \
  joint_lambda050_seed37_e51 \
  joint_lambda025_seed37_e51 \
  joint_lambda000_seed37_e51

echo "LAYER5_KAGGLE_PIPELINE_COMPLETE ${layer5_root}"
