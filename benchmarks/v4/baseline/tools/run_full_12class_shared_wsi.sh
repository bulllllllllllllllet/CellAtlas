#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CONFIG GPU START_TIMESTAMP LOG_PATH" >&2
  exit 2
fi

config=$1
gpu=$2
start_timestamp=$3
log_path=$4
task_stamps=(160000 160100 160200 160300 160400 160500 160600 122530 160700 160800 160900 161000)
case "$(basename "$config")" in
  sam_zero_shot.yaml) method=sam ;;
  sam_med2d_zero_shot.yaml) method=sam_med2d ;;
  wsi_sam_zero_shot.yaml) method=wsi_sam ;;
  *) echo "unsupported config name: $config" >&2; exit 2 ;;
esac

if [[ ! "$start_timestamp" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "START_TIMESTAMP must use YYYYMMDD_HHMMSS" >&2
  exit 2
fi

start_epoch=$(date -d "${start_timestamp:0:8} ${start_timestamp:9:2}:${start_timestamp:11:2}:${start_timestamp:13:2}" +%s)
exit_code=0
trap 'exit_code=$?; printf "EXIT_CODE=%s\\n" "$exit_code" >>"$log_path"' EXIT

cd /home/zhaoyh/CellAtlas
for index in "${!task_stamps[@]}"; do
  task_stamp=${task_stamps[$index]}
  run_stamp=$(date -d "@$((start_epoch + index * 60))" +%Y%m%d_%H%M%S)
  task_manifest="/nfs-medical3/zyh/v4/baseline/wsi_gt_guided_tasks_20260726_${task_stamp}/wsi_tasks_20260726_${task_stamp}.parquet"
  if [[ ! -f "$task_manifest" ]]; then
    echo "missing task manifest: $task_manifest" >&2
    exit 1
  fi
  output="/nfs-medical3/zyh/v4/baseline/${method}_gt_guided_wsi_${run_stamp}"
  if [[ -e "$output" ]]; then
    expected_wsi=$(conda run -n aligner python -c '
import sys
import pandas as pd
task = pd.read_parquet(sys.argv[1])
print(task["wsi_id"].nunique())
' "$task_manifest")
    completed_wsi=$(find "$output" -mindepth 2 -maxdepth 2 -name metadata.json -type f | wc -l)
    if [[ "$completed_wsi" -eq "$expected_wsi" ]]; then
      echo "validated completed output already exists; skipping: $output"
      continue
    fi
    echo "incomplete output already exists ($completed_wsi/$expected_wsi WSIs): $output" >&2
    exit 1
  fi
  conda run -n aligner python -m benchmarks.v4.baseline.tools.evaluate_gt_guided_wsi_patchwise \
    --config "$config" \
    --class-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
    --task-manifest "$task_manifest" \
    --gpu "$gpu" \
    --output-root /nfs-medical3/zyh/v4/baseline \
    --timestamp "$run_stamp" \
    --context-scale 2 \
    --context-prompt-mode all_in_context \
    >>"$log_path" 2>&1
done
