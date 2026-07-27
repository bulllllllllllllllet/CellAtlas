#!/usr/bin/env bash
# Wait for four truly idle GPUs, then launch the audited Phase-4 static-bidir run.
set -euo pipefail

required_gpus=4
poll_seconds=60
idle_minutes=10
utilization_max=10
minimum_free_mib=30720

if [[ $# -gt 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

started_stamp="$(date +%Y%m%d_%H%M%S)"
log_root="/nfs-medical3/zyh/v4/phase6/logs"
monitor_log="${log_root}/phase4_static_bidir_wait_${started_stamp}.log"
mkdir -p "$log_root"
exec > >(tee -a "$monitor_log") 2>&1

echo "monitor_started=${started_stamp}"
echo "required_gpus=${required_gpus} idle_minutes=${idle_minutes} utilization_max=${utilization_max} minimum_free_mib=${minimum_free_mib} shared_gpu_mode=true"

idle_polls_required=$((idle_minutes * 60 / poll_seconds))
idle_polls=0
selected=""

while true; do
    mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits)

    candidates=()
    for row in "${gpu_rows[@]}"; do
        IFS=',' read -r index util free_memory <<< "$row"
        index="${index// /}"; util="${util// /}"; free_memory="${free_memory// /}"
        if (( util > utilization_max || free_memory < minimum_free_mib )); then
            continue
        fi
        candidates+=("$index")
    done

    if (( ${#candidates[@]} >= required_gpus )); then
        selected="$(IFS=,; echo "${candidates[*]:0:required_gpus}")"
        idle_polls=$((idle_polls + 1))
        echo "$(date -Is) idle_candidate_gpus=${selected} consecutive_polls=${idle_polls}/${idle_polls_required}"
    else
        echo "$(date -Is) insufficient_idle_gpus=${#candidates[@]} candidates=$(IFS=,; echo "${candidates[*]:-none}")"
        idle_polls=0
        selected=""
    fi

    if (( idle_polls >= idle_polls_required )); then
        break
    fi
    sleep "$poll_seconds"
done

launch_stamp="$(date +%Y%m%d_%H%M%S)"
train_log="${log_root}/phase4_static_bidir_formal_${launch_stamp}.log"
echo "$(date -Is) launch_gpus=${selected} training_timestamp=${launch_stamp} training_log=${train_log}"

if CUDA_VISIBLE_DEVICES="$selected" conda run -n aligner torchrun --standalone --nproc_per_node="$required_gpus" \
    -m benchmarks.v4.phase_6_mask_decoder.train_joint_pixel \
    --config benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_phase4_static_bidir.yaml \
    --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
    --phase5-config benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml \
    --phase2-checkpoint /nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth \
    --cell-checkpoint /nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth \
    --phase5-checkpoint /nfs-medical3/zyh/v4/phase5/runs/phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth \
    --initial-joint-checkpoint /nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth \
    --cache-index /nfs-medical3/zyh/v4/phase4/data/multiscale_token_cache_20260719_205544/cache_index.parquet \
    --label-index /nfs-medical3/zyh/v4/phase4/data/fine_region_labels_20260719_230128/label_index.parquet \
    --patch-index /nfs-medical3/zyh/v4/phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet \
    --eligibility-index /nfs-medical3/zyh/v4/phase5/data/prompt_eligibility_20260720_210851/eligibility_index.parquet \
    --train-cell-routing /nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_train_20260719_014500_spatial/feature_routing.parquet \
    --val-cell-routing /nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_val_20260719_014500_spatial/feature_routing.parquet \
    --timestamp "$launch_stamp" \
    --log-every-steps 100 >"$train_log" 2>&1; then
    exit_code=0
else
    exit_code=$?
fi
printf '%s training_exit_code=%s\n' "$(date -Is)" "$exit_code"
printf 'EXIT_CODE=%s\n' "$exit_code" >> "$train_log"
exit "$exit_code"
