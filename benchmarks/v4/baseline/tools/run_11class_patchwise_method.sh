#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 CONFIG GPU RUN_HOUR" >&2
  exit 2
fi
config=$1
gpu=$2
run_hour=$3
cd /home/zhaoyh/CellAtlas
for ci in $(seq 0 10); do
  task_stamp=$(printf '20260726_16%02d00' "$ci")
  run_stamp=$(printf '20260726_%s%02d00' "$run_hour" "$ci")
  conda run -n aligner python -m benchmarks.v4.baseline.tools.evaluate_gt_guided_wsi_patchwise \
    --config "$config" \
    --class-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
    --task-manifest "/nfs-medical3/zyh/v4/baseline/wsi_gt_guided_tasks_${task_stamp}/wsi_tasks_${task_stamp}.parquet" \
    --gpu "$gpu" --output-root /nfs-medical3/zyh/v4/baseline --timestamp "$run_stamp"
done
