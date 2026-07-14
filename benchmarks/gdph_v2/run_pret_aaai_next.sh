#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/aligner/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_full10x_auto_physical}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_next_full10x}"
BASELINE_ROOT="${BASELINE_ROOT:-/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_full10x_auto_physical}"
WORKERS="${WORKERS:-4}"
IMAGE_ARGS=()

if [[ "${IMAGE_ID:-}" != "" ]]; then
  IMAGE_ARGS+=(--image_id "$IMAGE_ID")
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENCV_NUM_THREADS="${OPENCV_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cellatlas_mpl}"

mkdir -p "$OUTPUT_ROOT/logs"
cd "$ROOT_DIR"

echo "[$(date '+%F %T')] START next interaction/calibration"
"$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_next_experiments \
  --source_root "$SOURCE_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  "${IMAGE_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/logs/pret_aaai_next_experiments.log"
echo "[$(date '+%F %T')] DONE next interaction/calibration"

if [[ "${IMAGE_ID:-}" == "" ]]; then
  echo "[$(date '+%F %T')] START supervised upper bound"
  "$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_supervised_upper_bound \
    --source_root "$SOURCE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    2>&1 | tee "$OUTPUT_ROOT/logs/pret_aaai_supervised_upper_bound.log"
  echo "[$(date '+%F %T')] DONE supervised upper bound"
else
  echo "[$(date '+%F %T')] SKIP supervised upper bound for single-image smoke"
fi

echo "[$(date '+%F %T')] START report"
"$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_next_report \
  --output_root "$OUTPUT_ROOT" \
  --baseline_root "$BASELINE_ROOT" \
  2>&1 | tee "$OUTPUT_ROOT/logs/pret_aaai_next_report.log"
echo "[$(date '+%F %T')] DONE report"

echo "Report: $OUTPUT_ROOT/pret_superpixel/PRET_SUPERPIXEL_AAAI_NEXT.md"
