#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 TIMESTAMP CUDA_VISIBLE_DEVICES NPROC" >&2
    exit 2
fi

stamp="$1"
gpu_ids="$2"
nproc="$3"
log="/nfs-medical3/zyh/v4/phase6/logs/j5_full_budget_${stamp}.log"

record_exit() {
    task_code=$?
    printf '\nEXIT_CODE=%s\n' "$task_code"
}
trap record_exit EXIT
exec >"$log" 2>&1

CUDA_VISIBLE_DEVICES="$gpu_ids" conda run -n aligner torchrun \
    --standalone --nproc_per_node="$nproc" \
    -m benchmarks.v4.phase_6_mask_decoder.train_joint_pixel \
    --config benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_j5_full_budget.yaml \
    --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
    --phase5-config benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml \
    --phase2-checkpoint /nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth \
    --cell-checkpoint /nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth \
    --phase5-checkpoint /nfs-medical3/zyh/v4/phase5/runs/phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth \
    --initial-joint-checkpoint /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_030634/checkpoint_epoch_001.pth \
    --cache-index /nfs-medical3/zyh/v4/phase4/data/multiscale_token_cache_20260719_205544/cache_index.parquet \
    --label-index /nfs-medical3/zyh/v4/phase4/data/fine_region_labels_20260719_230128/label_index.parquet \
    --patch-index /nfs-medical3/zyh/v4/phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet \
    --eligibility-index /nfs-medical3/zyh/v4/phase5/data/prompt_eligibility_20260720_210851/eligibility_index.parquet \
    --train-cell-routing /nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_train_20260719_014500_spatial/feature_routing.parquet \
    --val-cell-routing /nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_val_20260719_014500_spatial/feature_routing.parquet \
    --timestamp "$stamp" \
    --log-every-steps 100
