#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 PID REQUEST_TIMESTAMP" >&2
    exit 2
fi

watched_pid="$1"
request_stamp="$2"
repo_root="/home/zhaoyh/CellAtlas"
log_root="/nfs-medical3/zyh/v4/phase6/logs"
gpu_ids="0,1,2,3,4,5"

cd "$repo_root" || exit 2
mkdir -p "$log_root"
watch_log="${log_root}/phase6_joint_wait6_${request_stamp}.log"
exec >>"$watch_log" 2>&1

record_exit() {
    task_code=$?
    printf '\nEXIT_CODE=%s\n' "$task_code"
}
trap record_exit EXIT

timestamp_message() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

timestamp_message "waiting for PID ${watched_pid}; request=${request_stamp}"
while kill -0 "$watched_pid" 2>/dev/null; do
    timestamp_message "PID ${watched_pid} is still running"
    sleep 60
done
timestamp_message "PID ${watched_pid} has exited; waiting for six stable GPUs"

stable_checks=0
while (( stable_checks < 3 )); do
    gpu_count=0
    all_ready=1
    gpu_snapshot=""
    while IFS=',' read -r gpu_index free_mib utilization; do
        gpu_index="${gpu_index// /}"
        free_mib="${free_mib// /}"
        utilization="${utilization// /}"
        [[ -z "$gpu_index" ]] && continue
        gpu_count=$((gpu_count + 1))
        gpu_snapshot+="gpu${gpu_index}:free=${free_mib}MiB,util=${utilization}% "
        if (( free_mib < 30000 || utilization > 10 )); then
            all_ready=0
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
    if (( gpu_count != 6 )); then
        all_ready=0
    fi
    if (( all_ready )); then
        stable_checks=$((stable_checks + 1))
    else
        stable_checks=0
    fi
    timestamp_message "GPU gate ${stable_checks}/3 ${gpu_snapshot}"
    if (( stable_checks < 3 )); then
        sleep 30
    fi
done

common_args=(
    --config benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_pixel.yaml
    --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml
    --phase5-config benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml
    --phase2-checkpoint /nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth
    --cell-checkpoint /nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth
    --phase5-checkpoint /nfs-medical3/zyh/v4/phase5/runs/phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth
    --cache-index /nfs-medical3/zyh/v4/phase4/data/multiscale_token_cache_20260719_205544/cache_index.parquet
    --label-index /nfs-medical3/zyh/v4/phase4/data/fine_region_labels_20260719_230128/label_index.parquet
    --patch-index /nfs-medical3/zyh/v4/phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet
    --eligibility-index /nfs-medical3/zyh/v4/phase5/data/prompt_eligibility_20260720_210851/eligibility_index.parquet
    --train-cell-routing /nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_train_20260719_014500_spatial/feature_routing.parquet
    --val-cell-routing /nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_val_20260719_014500_spatial/feature_routing.parquet
    --batch-size-per-gpu 2
    --num-workers 2
)

smoke_stamp="$(date '+%Y%m%d_%H%M%S')"
smoke_log="${log_root}/phase6_joint_6gpu_smoke_${smoke_stamp}.log"
timestamp_message "starting 6-GPU smoke; timestamp=${smoke_stamp}; log=${smoke_log}"
CUDA_VISIBLE_DEVICES="$gpu_ids" conda run -n aligner torchrun --standalone --nproc_per_node=6 \
    -m benchmarks.v4.phase_6_mask_decoder.train_joint_pixel \
    "${common_args[@]}" --timestamp "$smoke_stamp" --max-epochs 1 \
    --samples-per-epoch 600 --validation-samples 122 --log-every-steps 10 \
    >"$smoke_log" 2>&1
smoke_code=$?
printf '\nEXIT_CODE=%s\n' "$smoke_code" >>"$smoke_log"
timestamp_message "6-GPU smoke EXIT_CODE=${smoke_code}"
if (( smoke_code != 0 )); then
    timestamp_message "formal launch blocked by failed smoke"
    exit "$smoke_code"
fi

formal_stamp="$(date '+%Y%m%d_%H%M%S')"
formal_log="${log_root}/phase6_joint_6gpu_formal_${formal_stamp}.log"
timestamp_message "starting formal run; timestamp=${formal_stamp}; log=${formal_log}"
CUDA_VISIBLE_DEVICES="$gpu_ids" conda run -n aligner torchrun --standalone --nproc_per_node=6 \
    -m benchmarks.v4.phase_6_mask_decoder.train_joint_pixel \
    "${common_args[@]}" --timestamp "$formal_stamp" --log-every-steps 100 \
    >"$formal_log" 2>&1
formal_code=$?
printf '\nEXIT_CODE=%s\n' "$formal_code" >>"$formal_log"
timestamp_message "formal run EXIT_CODE=${formal_code}"
exit "$formal_code"
