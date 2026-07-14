#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  benchmarks/gdph_v2/run_pret_superpixel_scale.sh [options]

Options:
  --scale 4096|8192|full10x     Default: 8192
  --output-root PATH            Override output root
  --source-root PATH            Existing GDPH v2 root with cells/masks/patches/manifests/region_retrieval
  --manifest PATH               Manifest CSV. Default: SOURCE_ROOT/manifests/main_20.csv
  --token-workers N             Default: 6
  --eval-workers N              Default: 8
  --target-sp-diameter N|auto_physical
                                  Superpixel target diameter. full10x defaults to auto_physical.
                                  auto_physical = 64 * current_max_dimension / 4096 per slide.
  --canonical-prompts PATH      Restrict prompts to a scale-stable canonical prompt CSV
  --compare-with NAME=ROOT      After run, compare current output with one or more previous runs
  --skip-tokens                 Reuse existing pret_superpixel/<image_id> token outputs
  --skip-prompts                Reuse existing pret_superpixel/prompts.csv
  --skip-visualize              Skip visualization generation
  --with-oracle-negative        Also generate oracle_positive_negative prompts
  -h, --help                    Show this help

Default run:
  main20 8192, core variants, deployable thresholds, random/shuffled sanity baselines.
EOF
}

SCALE="8192"
SOURCE_ROOT="/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2"
OUTPUT_ROOT=""
MANIFEST=""
TOKEN_WORKERS=6
EVAL_WORKERS=8
TARGET_SP_DIAMETER=64
TARGET_SP_DIAMETER_SET=0
SKIP_TOKENS=0
SKIP_PROMPTS=0
SKIP_VISUALIZE=0
NEGATIVE_MODE="none"
CANONICAL_PROMPTS=""
COMPARE_WITH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)
      SCALE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --token-workers)
      TOKEN_WORKERS="$2"
      shift 2
      ;;
    --eval-workers)
      EVAL_WORKERS="$2"
      shift 2
      ;;
    --target-sp-diameter)
      TARGET_SP_DIAMETER="$2"
      TARGET_SP_DIAMETER_SET=1
      shift 2
      ;;
    --canonical-prompts)
      CANONICAL_PROMPTS="$2"
      shift 2
      ;;
    --compare-with)
      COMPARE_WITH+=("$2")
      shift 2
      ;;
    --skip-tokens)
      SKIP_TOKENS=1
      shift
      ;;
    --skip-prompts)
      SKIP_PROMPTS=1
      shift
      ;;
    --skip-visualize)
      SKIP_VISUALIZE=1
      shift
      ;;
    --with-oracle-negative)
      NEGATIVE_MODE="oracle_contrast"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$SCALE" in
  4096)
    MAX_TARGET_DIMENSION=4096
    DEFAULT_OUTPUT_ROOT="/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_4096_thresholds"
    ;;
  8192)
    MAX_TARGET_DIMENSION=8192
    DEFAULT_OUTPUT_ROOT="/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_8192"
    ;;
  full10x)
    MAX_TARGET_DIMENSION=0
    DEFAULT_OUTPUT_ROOT="/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_full10x"
    if [[ "$TARGET_SP_DIAMETER_SET" -eq 0 ]]; then
      TARGET_SP_DIAMETER="auto_physical"
    fi
    ;;
  *)
    echo "--scale must be one of: 4096, 8192, full10x" >&2
    exit 2
    ;;
esac

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
fi
if [[ -z "$MANIFEST" ]]; then
  MANIFEST="$SOURCE_ROOT/manifests/main_20.csv"
fi

PYTHON="${PYTHON:-$HOME/miniconda3/envs/aligner/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENCV_NUM_THREADS="${OPENCV_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cellatlas_mpl}"

run_logged() {
  local step_name="$1"
  local log_path="$2"
  shift 2
  echo
  echo "[$(date '+%F %T')] START $step_name"
  echo "  log=$log_path"
  "$@" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED $step_name status=$status"
    return "$status"
  fi
  echo "[$(date '+%F %T')] DONE $step_name"
}

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/pret_superpixel"
for name in cells masks patches manifests region_retrieval; do
  if [[ -e "$SOURCE_ROOT/$name" ]]; then
    ln -sfn "$SOURCE_ROOT/$name" "$OUTPUT_ROOT/$name"
  fi
done

echo "PRET superpixel scale run"
echo "  scale=$SCALE"
echo "  output_root=$OUTPUT_ROOT"
echo "  source_root=$SOURCE_ROOT"
echo "  manifest=$MANIFEST"
echo "  max_target_dimension=$MAX_TARGET_DIMENSION"
echo "  target_sp_diameter=$TARGET_SP_DIAMETER"
echo "  token_workers=$TOKEN_WORKERS eval_workers=$EVAL_WORKERS"
if [[ -n "$CANONICAL_PROMPTS" ]]; then
  echo "  canonical_prompts=$CANONICAL_PROMPTS"
fi

if [[ "$SKIP_TOKENS" -eq 0 ]]; then
  run_logged "tokens scale=$SCALE" "$OUTPUT_ROOT/logs/pret_superpixel_tokens_${SCALE}.log" \
  "$PYTHON" -u -m benchmarks.gdph_v2.pret_superpixel_tokens \
    --manifest "$MANIFEST" \
    --output_root "$OUTPUT_ROOT" \
    --workers "$TOKEN_WORKERS" \
    --max_target_dimension "$MAX_TARGET_DIMENSION" \
    --target_sp_diameter "$TARGET_SP_DIAMETER" \
    --base_max_dimension 4096 \
    --base_sp_diameter 64
else
  echo "Skipping token generation"
fi

if [[ "$SKIP_PROMPTS" -eq 0 ]]; then
  PROMPT_ARGS=(
    --output_root "$OUTPUT_ROOT"
    --negative_mode "$NEGATIVE_MODE"
  )
  if [[ -n "$CANONICAL_PROMPTS" ]]; then
    PROMPT_ARGS+=(--canonical_prompts_csv "$CANONICAL_PROMPTS")
  fi
  run_logged "prompts scale=$SCALE" "$OUTPUT_ROOT/logs/pret_generate_prompts_${SCALE}.log" \
  "$PYTHON" -u -m benchmarks.gdph_v2.pret_generate_prompts "${PROMPT_ARGS[@]}"
else
  echo "Skipping prompt generation"
fi

run_logged "eval scale=$SCALE" "$OUTPUT_ROOT/logs/pret_eval_${SCALE}.log" \
"$PYTHON" -u -m benchmarks.gdph_v2.pret_eval_in_context \
  --output_root "$OUTPUT_ROOT" \
  --variants image_only cell_reg image_cell_reg_cellw0p25 image_cell_reg_cellw0p5 random_token \
  --baselines none random_prompt shuffled_token \
  --smoothing_alphas 0 \
  --area_ratios 0.01 0.02 0.05 0.10 0.15 0.18 0.20 0.30 \
  --workers "$EVAL_WORKERS"

run_logged "analyze scale=$SCALE" "$OUTPUT_ROOT/logs/pret_analyze_${SCALE}.log" \
"$PYTHON" -u -m benchmarks.gdph_v2.pret_analyze_results \
  --output_root "$OUTPUT_ROOT" \
  --primary_variant image_cell_reg_cellw0p5

if [[ "$SKIP_VISUALIZE" -eq 0 ]]; then
  : > "$OUTPUT_ROOT/logs/pret_visualize_${SCALE}.log"
  for class_id in 3 6 8; do
    run_logged "visualize class=$class_id scale=$SCALE" "$OUTPUT_ROOT/logs/pret_visualize_${SCALE}_class${class_id}.log" \
    "$PYTHON" -u -m benchmarks.gdph_v2.pret_visualize \
      --output_root "$OUTPUT_ROOT" \
      --variant image_cell_reg_cellw0p5 \
      --class_id "$class_id" \
      --prompt_source realistic_box \
      --scope exclude_prompt_region \
      --mask_protocol percentile_90 \
      --selection best median worst \
      --limit_per_class 1 \
      --max_visualizations 6
    cat "$OUTPUT_ROOT/logs/pret_visualize_${SCALE}_class${class_id}.log" >> "$OUTPUT_ROOT/logs/pret_visualize_${SCALE}.log"
  done
else
  echo "Skipping visualization"
fi

if [[ "${#COMPARE_WITH[@]}" -gt 0 ]]; then
  COMPARE_ARGS=("${COMPARE_WITH[@]}" "$SCALE=$OUTPUT_ROOT")
  run_logged "compare scale=$SCALE" "$OUTPUT_ROOT/logs/pret_compare_${SCALE}.log" \
  "$PYTHON" -u -m benchmarks.gdph_v2.pret_compare_scales \
    --runs "${COMPARE_ARGS[@]}" \
    --output_dir "$OUTPUT_ROOT/pret_scale_comparison"
fi

echo "Done."
echo "Report: $OUTPUT_ROOT/pret_superpixel/PRET_SUPERPIXEL_ANALYSIS.md"
echo "Metrics: $OUTPUT_ROOT/pret_superpixel/pret_metrics.csv"
echo "Validation: $OUTPUT_ROOT/pret_superpixel/pret_eval_validation.json"
