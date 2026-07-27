#!/usr/bin/env bash
set -euo pipefail

stamp="${1:?usage: wait_launch_prompt_mean_smoke.sh YYYYMMDD_HHMMSS}"
gpu_ids="1,3,4,5"
idle_checks=0

while (( idle_checks < 3 )); do
  busy_count=$(nvidia-smi --id="$gpu_ids" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1 > 5 || $2 > 500) busy++} END {print busy+0}')
  if [[ "$busy_count" == "0" ]]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  printf '%s gpu_ids=%s busy_count=%s consecutive_idle_checks=%s/3\n' "$(date '+%F %T')" "$gpu_ids" "$busy_count" "$idle_checks"
  if (( idle_checks < 3 )); then
    sleep 60
  fi
done

cd /home/zhaoyh/CellAtlas
CUDA_VISIBLE_DEVICES="$gpu_ids" conda run -n aligner torchrun --standalone --nproc_per_node=4 \
  -m benchmarks.v4.phase_6_mask_decoder.train_joint_pixel \
  --config benchmarks/v4/ablation/configs/prompt_mean_prototype.yaml \
  --initial-joint-checkpoint /nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth \
  --timestamp "$stamp" \
  --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
  --phase5-config benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml \
  --phase2-checkpoint /nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth \
  --cell-checkpoint /nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth \
  --phase5-checkpoint /nfs-medical3/zyh/v4/phase5/runs/phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth \
  --cache-index /nfs-medical3/zyh/v4/phase4/data/multiscale_token_cache_20260719_205544/cache_index.parquet \
  --label-index /nfs-medical3/zyh/v4/phase4/data/fine_region_labels_20260719_230128/label_index.parquet \
  --patch-index /nfs-medical3/zyh/v4/phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet \
  --eligibility-index /nfs-medical3/zyh/v4/phase5/data/prompt_eligibility_20260720_210851/eligibility_index.parquet \
  --train-cell-routing /nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_train_20260719_014500_spatial/feature_routing.parquet \
  --val-cell-routing /nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_val_20260719_014500_spatial/feature_routing.parquet \
  --max-epochs 1 --samples-per-epoch 32 --validation-samples 64
