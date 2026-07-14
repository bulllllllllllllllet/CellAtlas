#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/aligner/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_full10x_auto_physical}"
AAAI_ROOT="${AAAI_ROOT:-/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_full10x_auto_physical}"
NEXT_ROOT="${NEXT_ROOT:-/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v2_full10x}"
PRIMARY_VARIANT="${PRIMARY_VARIANT:-image_cell_reg_cellw0p5}"
RUN_ENHANCE_TOKENS="${RUN_ENHANCE_TOKENS:-0}"
WORKERS="${WORKERS:-4}"
VIS_LIMIT_PER_CLASS="${VIS_LIMIT_PER_CLASS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENCV_NUM_THREADS="${OPENCV_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cellatlas_mpl}"

mkdir -p "$NEXT_ROOT/logs"
cd "$ROOT_DIR"

if [[ "$RUN_ENHANCE_TOKENS" == "1" ]]; then
  echo "[$(date '+%F %T')] START enhanced token generation"
  "$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_enhance_tokens \
    --source_root "$SOURCE_ROOT" \
    --workers "$WORKERS" \
    2>&1 | tee "$NEXT_ROOT/logs/pret_aaai_enhance_tokens.log"
  echo "[$(date '+%F %T')] DONE enhanced token generation"
fi

echo "[$(date '+%F %T')] START next experiments with strict hard negatives"
"$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_next_experiments \
  --source_root "$SOURCE_ROOT" \
  --output_root "$NEXT_ROOT" \
  --workers "$WORKERS" \
  --primary_variant "$PRIMARY_VARIANT" \
  2>&1 | tee "$NEXT_ROOT/logs/pret_aaai_next_experiments_final.log"
echo "[$(date '+%F %T')] DONE next experiments"

echo "[$(date '+%F %T')] START supervised upper bound"
"$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_supervised_upper_bound \
  --source_root "$SOURCE_ROOT" \
  --output_root "$NEXT_ROOT" \
  --workers "$WORKERS" \
  2>&1 | tee "$NEXT_ROOT/logs/pret_aaai_supervised_upper_bound_final.log"
echo "[$(date '+%F %T')] DONE supervised upper bound"

echo "[$(date '+%F %T')] START next report"
"$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_next_report \
  --output_root "$NEXT_ROOT" \
  --baseline_root "$AAAI_ROOT" \
  2>&1 | tee "$NEXT_ROOT/logs/pret_aaai_next_report_final.log"
echo "[$(date '+%F %T')] DONE next report"

echo "[$(date '+%F %T')] START final tables/results"
"$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_final_outputs \
  --next_root "$NEXT_ROOT" \
  --aaai_root "$AAAI_ROOT" \
  --source_root "$SOURCE_ROOT" \
  2>&1 | tee "$NEXT_ROOT/logs/pret_aaai_final_outputs.log"
echo "[$(date '+%F %T')] DONE final tables/results"

echo "[$(date '+%F %T')] START visual summary"
"$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_visual_summary \
  --next_root "$NEXT_ROOT" \
  --source_root "$SOURCE_ROOT" \
  --limit_per_class "$VIS_LIMIT_PER_CLASS" \
  2>&1 | tee "$NEXT_ROOT/logs/pret_aaai_visual_summary.log"
echo "[$(date '+%F %T')] DONE visual summary"

echo "Final main table: $NEXT_ROOT/pret_superpixel/final_main_table.md"
echo "Query-weighted table: $NEXT_ROOT/pret_superpixel/final_main_table_query_weighted.md"
echo "Class-balanced table: $NEXT_ROOT/pret_superpixel/final_main_table_class_balanced.md"
echo "AAAI results: $NEXT_ROOT/pret_superpixel/AAAI_RESULTS.md"
echo "Visual summary: $NEXT_ROOT/pret_superpixel/visual_summary"
