#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 5 && $# -ne 6 ]]; then
  echo "usage: $0 CONFIG GPU TIMESTAMP WSI_ID LOG_PATH [LIMIT_CALLS]" >&2
  exit 2
fi

config=$1
gpu=$2
timestamp=$3
wsi_id=$4
log_path=$5
limit_args=()
if [[ $# -eq 6 ]]; then
  limit_args=(--limit-calls "$6")
fi

cd /home/zhaoyh/CellAtlas
conda run -n aligner python -m benchmarks.v4.baseline.tools.evaluate_gt_guided_wsi_patchwise \
  --config "$config" \
  --class-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
  --task-manifest /nfs-medical3/zyh/v4/baseline/wsi_gt_guided_tasks_20260726_122530/wsi_tasks_20260726_122530.parquet \
  --wsi-id "$wsi_id" \
  --gpu "$gpu" \
  --output-root /nfs-medical3/zyh/v4/baseline \
  --timestamp "$timestamp" \
  --context-scale 2 \
  --context-prompt-mode all_in_context \
  "${limit_args[@]}" \
  >>"$log_path" 2>&1
exit_code=$?
printf 'EXIT_CODE=%s\n' "$exit_code" >>"$log_path"
exit "$exit_code"
