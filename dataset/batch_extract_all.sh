#!/bin/bash
# Batch extraction: 2-pass pipeline for all CRC datasets
# Pass 1: extract_features_wsi_stream.py (Cellpose + HE hard-mask + mIF mask-avg)
# Pass 2: re_extract_crc02.py --mif_path (HE centered-crop + mIF mask-avg from TIFF)
#
# Usage: bash dataset/batch_extract_all.sh [dataset_name]
#   dataset_name: optional, if provided only processes that one dataset
set -e

DATA_ROOT="/nfs-medical3/yjz/OrionCRC"
CACHE_ROOT="dataset/crc_cache"
CTP_WEIGHTS="module/checkpoint/ctranspath.pth"
LOG_DIR="logs"

# All CRC datasets
ALL_DATASETS=(
  CRC01 CRC02 CRC03 CRC04 CRC05 CRC06 CRC07 CRC08 CRC09 CRC10
  CRC11 CRC12 CRC13 CRC14 CRC15 CRC16 CRC17 CRC18 CRC19 CRC20
  CRC21 CRC22 CRC23 CRC24 CRC25 CRC26 CRC27 CRC28 CRC29 CRC30
  CRC31 CRC32 CRC33_01 CRC33_02 CRC34 CRC35 CRC36 CRC37 CRC38
  CRC39 CRC40
)

if [ $# -ge 1 ]; then
  DATASETS=("$1")
  echo "Single-dataset mode: $1"
else
  DATASETS=("${ALL_DATASETS[@]}")
  echo "Batch mode: ${#DATASETS[@]} datasets"
fi

export CUDA_VISIBLE_DEVICES=1

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "============================================"
  echo "[$(date '+%H:%M:%S')] Processing $dataset"
  echo "============================================"

  HE=$(ls "$DATA_ROOT/$dataset/"*registered.ome.tif)
  MIF=$(ls "$DATA_ROOT/$dataset/"*zlib.ome.tiff)
  PASS1_DIR="$CACHE_ROOT/$dataset/pass1"
  FINAL_DIR="$CACHE_ROOT/$dataset"

  # Skip if final output already exists
  if [ -f "$FINAL_DIR/global_norm_stats.json" ]; then
    echo "  Final cache exists, skipping."
    continue
  fi

  # === Pass 1: Cellpose + initial feature extraction ===
  if [ ! -d "$PASS1_DIR/he" ] || [ $(ls "$PASS1_DIR/he" 2>/dev/null | wc -l) -eq 0 ]; then
    echo "  [Pass 1] Cellpose + feature extraction..."
    python extract_features_wsi_stream.py \
      --he "$HE" --mif "$MIF" \
      --cache_dir "$PASS1_DIR" \
      --log_dir "$LOG_DIR" \
      --weights "$CTP_WEIGHTS" \
      --num_workers 1

    echo "  [Pass 1] Complete: $(ls $PASS1_DIR/he 2>/dev/null | wc -l) patches"
  else
    echo "  [Pass 1] Already exists, skipping."
  fi

  # Verify pass 1 produced output
  N_PASS1=$(ls "$PASS1_DIR/he" 2>/dev/null | wc -l)
  if [ "$N_PASS1" -eq 0 ]; then
    echo "  ERROR: Pass 1 produced 0 patches. Skipping $dataset."
    continue
  fi

  # === Pass 2: Re-extract HE (centered-crop) + mIF (mask averaging) ===
  N_FINAL=$(ls "$FINAL_DIR/he" 2>/dev/null | wc -l)
  if [ ! -d "$FINAL_DIR/he" ] || [ "$N_FINAL" -lt "$N_PASS1" ]; then
    echo "  [Pass 2] Re-extracting HE (centered-crop) + mIF (mask-averaging)..."
    python dataset/re_extract_crc02.py \
      --input_dir "$PASS1_DIR" \
      --output_dir "$FINAL_DIR" \
      --mif_path "$MIF" \
      --weights "$CTP_WEIGHTS"

    echo "  [Pass 2] Complete: $(ls $FINAL_DIR/he 2>/dev/null | wc -l) patches"
  else
    echo "  [Pass 2] Already complete ($N_FINAL/$N_PASS1 patches), skipping."
  fi

  # Verify pass 2 produced enough output
  N_FINAL=$(ls "$FINAL_DIR/he" 2>/dev/null | wc -l)
  if [ "$N_FINAL" -lt "$N_PASS1" ]; then
    echo "  WARNING: Pass 2 only has $N_FINAL/$N_PASS1 patches from pass 1."
    echo "  Pass 1 may have been partial. Proceeding anyway with $N_FINAL patches."
  fi

  # Clean up pass 1 to save disk
  echo "  [Cleanup] Removing pass1 cache..."
  rm -rf "$PASS1_DIR"

  # Compute global_norm_stats
  echo "  [Stats] Computing global_norm_stats..."
  python compute_global_norm_stats.py --cache_dir "$FINAL_DIR"

  echo "  Done: $dataset ($N_FINAL patches)"
done

echo ""
echo "============================================"
echo "All datasets processed!"
echo "============================================"
echo ""
echo "Cache directories:"
for dataset in "${DATASETS[@]}"; do
  echo "  $CACHE_ROOT/$dataset  ($(ls $CACHE_ROOT/$dataset/he 2>/dev/null | wc -l) patches)"
done
