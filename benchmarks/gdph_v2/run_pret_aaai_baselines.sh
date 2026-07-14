#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_full10x_auto_physical"
OUTPUT_ROOT="/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_full10x_auto_physical"
PYTHON="${HOME}/miniconda3/envs/aligner/bin/python"
WORKERS=4
RUN_SAM=1
RUN_MEDSAM=1
SAM_CHECKPOINT="/nfs-medical3/zyh/models/sam/sam_vit_b_01ec64.pth"
MEDSAM_CHECKPOINT="/nfs-medical3/zyh/models/medsam/medsam_vit_b.pth"

usage() {
  cat <<'USAGE'
Usage: run_pret_aaai_baselines.sh [options]

Options:
  --source-root PATH       Existing full10x auto-physical PRET root
  --output-root PATH       AAAI output root
  --python PATH            Python executable
  --workers N              PRET eval workers
  --sam-checkpoint PATH    SAM ViT-B checkpoint
  --medsam-checkpoint PATH MedSAM ViT-B checkpoint
  --skip-sam               Do not run SAM baseline
  --skip-medsam            Do not run MedSAM baseline
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --sam-checkpoint) SAM_CHECKPOINT="$2"; shift 2 ;;
    --medsam-checkpoint) MEDSAM_CHECKPOINT="$2"; shift 2 ;;
    --skip-sam) RUN_SAM=0; shift ;;
    --skip-medsam) RUN_MEDSAM=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

PRET_OUT="${OUTPUT_ROOT}/pret_superpixel"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$PRET_OUT" "$LOG_DIR"

for name in cells masks patches manifests region_retrieval; do
  if [[ -e "${SOURCE_ROOT}/${name}" && ! -e "${OUTPUT_ROOT}/${name}" ]]; then
    ln -s "${SOURCE_ROOT}/${name}" "${OUTPUT_ROOT}/${name}"
  fi
done

for item in "${SOURCE_ROOT}"/pret_superpixel/*; do
  base="$(basename "$item")"
  case "$base" in
    pret_aaai_*|PRET_SUPERPIXEL_AAAI_BASELINE.md|sam_vit_b_*|medsam_vit_b_*|visualizations|aaai_visualizations)
      continue
      ;;
  esac
  if [[ ! -e "${PRET_OUT}/${base}" ]]; then
    ln -s "$item" "${PRET_OUT}/${base}"
  fi
done

echo "[AAAI] source=${SOURCE_ROOT}"
echo "[AAAI] output=${OUTPUT_ROOT}"

"$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_eval \
  --source_root "$SOURCE_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  2>&1 | tee "${LOG_DIR}/pret_aaai_eval.log"

if [[ "$RUN_SAM" == "1" ]]; then
  "$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_sam_baseline \
    --source_root "$SOURCE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --checkpoint "$SAM_CHECKPOINT" \
    --model_type vit_b \
    --model_name sam_vit_b \
    --workers 1 \
    2>&1 | tee "${LOG_DIR}/sam_vit_b_baseline.log"
fi

if [[ "$RUN_MEDSAM" == "1" ]]; then
  "$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_sam_baseline \
    --source_root "$SOURCE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --checkpoint "$MEDSAM_CHECKPOINT" \
    --model_type vit_b \
    --model_name medsam_vit_b \
    --workers 1 \
    2>&1 | tee "${LOG_DIR}/medsam_vit_b_baseline.log"
fi

"$PYTHON" -u -m benchmarks.gdph_v2.pret_aaai_report \
  --output_root "$OUTPUT_ROOT" \
  2>&1 | tee "${LOG_DIR}/pret_aaai_report.log"

for class_id in 0 1 3 4 6 7 8 9 10; do
  "$PYTHON" -u -m benchmarks.gdph_v2.pret_visualize \
    --output_root "$OUTPUT_ROOT" \
    --variant image_cell_reg_cellw0p5 \
    --class_id "$class_id" \
    --prompt_source realistic_box \
    --scope exclude_prompt_region \
    --mask_protocol percentile_90 \
    --selection best median worst \
    --limit_per_class 1 \
    --max_visualizations 3 \
    --output_mode folder \
    --max_render_dimension 4096 \
    2>&1 | tee "${LOG_DIR}/pret_visualize_class${class_id}.log"
done

if [[ ! -e "${PRET_OUT}/aaai_visualizations" ]]; then
  ln -s "${PRET_OUT}/visualizations" "${PRET_OUT}/aaai_visualizations"
fi

echo "[AAAI] done"
echo "Report: ${PRET_OUT}/PRET_SUPERPIXEL_AAAI_BASELINE.md"
