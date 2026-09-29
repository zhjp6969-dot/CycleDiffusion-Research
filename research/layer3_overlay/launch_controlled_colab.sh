#!/usr/bin/env bash
set -euo pipefail

variant="${1:-uniform}"
seed="${2:-37}"
project_root="${CYCLEDIFFUSION_ROOT:-/content/CycleDiffusion}"
results_root="${CYCLEDIFFUSION_RESULTS_ROOT:-/content/drive/MyDrive/CycleDiffusion_results/layers3_5/full4_controlled}"
start_epoch="${START_EPOCH:-51}"
end_epoch="${END_EPOCH:-51}"
output_dir="${results_root}/${variant}_seed${seed}_e${end_epoch}"
state_path="${output_dir}/training_state_latest.pt"
final_checkpoint="${output_dir}/vc_${end_epoch}.pt"

case "${variant}" in
  uniform|speaker_only|content_only|joint|hard_gate) ;;
  *)
    echo "unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

required=(
  "${project_root}/train_reliability_cycle.py"
  "${project_root}/data_reliability.py"
  "${project_root}/VCTK_2F2M"
  "${project_root}/checkpts/spk_encoder/enc.pt"
  "${project_root}/real_last_cycle_train_dec_4speakers_original/vc_50_0823.pt"
  "${project_root}/artifacts/train_centroids.npz"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required asset: ${path}" >&2
    exit 3
  fi
done

mkdir -p "${output_dir}"
if [[ -f "${final_checkpoint}" ]]; then
  echo "already complete: ${final_checkpoint}"
  exit 0
fi

resume_args=()
if [[ -f "${state_path}" ]]; then
  resume_args=(--resume-training-state "${state_path}")
  echo "resuming from ${state_path}"
else
  echo "starting fresh controlled run in ${output_dir}"
fi

exec python -u "${project_root}/train_reliability_cycle.py" \
  --data-dir "${project_root}/VCTK_2F2M" \
  --encoder-checkpoint "${project_root}/checkpts/spk_encoder/enc.pt" \
  --resume "${project_root}/real_last_cycle_train_dec_4speakers_original/vc_50_0823.pt" \
  --centroids "${project_root}/artifacts/train_centroids.npz" \
  --output-dir "${output_dir}" \
  --variant "${variant}" \
  --lambda-cycle 1 \
  --start-epoch "${start_epoch}" \
  --end-epoch "${end_epoch}" \
  --checkpoint-every 1 \
  --checkpoint-every-steps 25 \
  --batch-size 4 \
  --cycle-batch-size 3 \
  --diffusion-steps 6 \
  --learning-rate 3e-5 \
  --seed "${seed}" \
  "${resume_args[@]}"
