#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 WAIT_PID EXPECTED_SUBSTRING MANIFEST CONFIG BATCH_ROOT TIMESTAMP" >&2
  exit 2
fi

wait_pid="$1"
expected_substring="$2"
manifest="$3"
config="$4"
batch_root="$5"
timestamp="$6"
repo_root="/home/zhaoyh/CellAtlas"

echo "{\"event\":\"wait_started\",\"wait_pid\":${wait_pid},\"expected_substring\":\"${expected_substring}\",\"timestamp\":\"$(date -Iseconds)\"}"
while [[ -r "/proc/${wait_pid}/cmdline" ]]; do
  target_cmd="$(tr '\0' ' ' < "/proc/${wait_pid}/cmdline")"
  if [[ "${target_cmd}" != *"${expected_substring}"* ]]; then
    echo "{\"event\":\"pid_identity_changed\",\"wait_pid\":${wait_pid},\"timestamp\":\"$(date -Iseconds)\"}"
    break
  fi
  echo "{\"event\":\"still_waiting\",\"wait_pid\":${wait_pid},\"timestamp\":\"$(date -Iseconds)\"}"
  sleep 60
done

echo "{\"event\":\"predecessor_exited\",\"wait_pid\":${wait_pid},\"timestamp\":\"$(date -Iseconds)\"}"
cd "${repo_root}"
set +e
conda run --no-capture-output -n aligner \
  python -m benchmarks.v4.phase_3_cell_region.tools.run_test_wsi_feature_batch \
  --manifest "${manifest}" \
  --config "${config}" \
  --batch-root "${batch_root}" \
  --timestamp "${timestamp}" \
  --gpus 0 1 2 3 4 5 \
  --shard-size 25 \
  --cell-batch-size 255 \
  --max-cells 255 \
  --spatial-grid-size 8
exit_code=$?
set -e
echo "{\"event\":\"successor_exited\",\"exit_code\":${exit_code},\"timestamp\":\"$(date -Iseconds)\"}"
exit "${exit_code}"
