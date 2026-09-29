#!/usr/bin/env bash
set -euo pipefail

# Clean independent-cohort confirmation. The three base models start from the
# archived acoustic encoder and random decoder weights; no discovery-cohort VC
# checkpoint is loaded. Each candidate/control pair branches from the same
# seed-matched base checkpoint.

project_root="${CYCLEDIFFUSION_ROOT:-/kaggle/input/datasets/jinpeng12353/cyclediffusion-private-research-assets/CycleDiffusion_kaggle_assets/CycleDiffusion}"
overlay_root="${LAYER10_OVERLAY_ROOT:-/kaggle/working/layer10_overlay}"
data_root="${LAYER10_DATA_ROOT:-/kaggle/working/layer10_cohort}"
results_root="${LAYER10_RESULTS_ROOT:-/kaggle/working/layer10_training}"
base_epochs="${LAYER10_BASE_EPOCHS:-50}"
continuation_steps="${LAYER10_CONTINUATION_STEPS:-461}"

required=(
  "${overlay_root}/train_reliability_cycle.py"
  "${overlay_root}/data_reliability.py"
  "${overlay_root}/reliability_cycle.py"
  "${overlay_root}/build_training_centroids.py"
  "${project_root}/checkpts/spk_encoder/enc.pt"
  "${data_root}/mels"
  "${data_root}/embeds"
  "${data_root}/preprocess_summary.json"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required Layer 10 asset: ${path}" >&2
    exit 3
  fi
done

export PYTHONPATH="${overlay_root}:${project_root}:${PYTHONPATH:-}"
mkdir -p "${results_root}/artifacts"
centroids="${results_root}/artifacts/train_centroids.npz"
if [[ ! -f "${centroids}" ]]; then
  python -u "${overlay_root}/build_training_centroids.py" \
    --embed-dir "${data_root}/embeds" \
    --output "${centroids}"
fi

run_base() {
  local seed="$1"
  local output_dir="${results_root}/seed${seed}/base"
  local final_checkpoint="${output_dir}/vc_${base_epochs}.pt"
  mkdir -p "${output_dir}"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "base already complete: ${final_checkpoint}"
    return
  fi
  local resume_args=()
  if [[ -f "${output_dir}/training_state_latest.pt" ]]; then
    resume_args=(--resume-training-state "${output_dir}/training_state_latest.pt")
  fi
  python -u "${overlay_root}/train_reliability_cycle.py" \
    --data-dir "${data_root}" \
    --encoder-checkpoint "${project_root}/checkpts/spk_encoder/enc.pt" \
    --centroids "${centroids}" \
    --output-dir "${output_dir}" \
    --variant uniform \
    --lambda-cycle 0 \
    --start-epoch 1 \
    --end-epoch "${base_epochs}" \
    --checkpoint-every 10 \
    --checkpoint-every-steps 100 \
    --batch-size 4 \
    --cycle-batch-size 3 \
    --diffusion-steps 6 \
    --learning-rate 3e-5 \
    --seed "${seed}" \
    "${resume_args[@]}"
}

run_branch() {
  local seed="$1"
  local variant="$2"
  local output_dir="${results_root}/seed${seed}/${variant}_lambda100"
  local base_checkpoint="${results_root}/seed${seed}/base/vc_${base_epochs}.pt"
  local final_checkpoint="${output_dir}/vc_step${continuation_steps}.pt"
  mkdir -p "${output_dir}"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "branch already complete: ${final_checkpoint}"
    return
  fi
  local resume_args=()
  if [[ -f "${output_dir}/training_state_latest.pt" ]]; then
    resume_args=(--resume-training-state "${output_dir}/training_state_latest.pt")
  fi
  python -u "${overlay_root}/train_reliability_cycle.py" \
    --data-dir "${data_root}" \
    --encoder-checkpoint "${project_root}/checkpts/spk_encoder/enc.pt" \
    --resume "${base_checkpoint}" \
    --centroids "${centroids}" \
    --output-dir "${output_dir}" \
    --variant "${variant}" \
    --lambda-cycle 1 \
    --start-epoch 51 \
    --end-epoch 52 \
    --max-total-steps "${continuation_steps}" \
    --checkpoint-every 10 \
    --checkpoint-every-steps 25 \
    --batch-size 4 \
    --cycle-batch-size 3 \
    --diffusion-steps 6 \
    --learning-rate 3e-5 \
    --seed "${seed}" \
    "${resume_args[@]}"
}

for seed in 17 37 73; do
  run_base "${seed}"
  run_branch "${seed}" uniform
  run_branch "${seed}" content_only
done

echo "LAYER10_TRAINING_COMPLETE ${results_root}"
