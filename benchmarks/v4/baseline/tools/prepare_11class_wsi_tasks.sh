#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaoyh/CellAtlas

classes=(tumor_epithelium tumor_stroma background necrosis normal_gland normal_stroma submucosa_serosa lymphocyte_aggregate mucus fat blood)
tiles=(
  /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1033746-12-HE-DX1_20260725_194443/wsi_tile_index.parquet
  /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1028417-R1-HE-DX1_20260725_200000/wsi_tile_index.parquet
  /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1321593-10-HE-DX1_20260725_200000/wsi_tile_index.parquet
  /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1416664-10-HE-DX1_20260725_200000/wsi_tile_index.parquet
  /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1504774-12-HE-DX1_20260725_200000/wsi_tile_index.parquet
)

for index in "${!classes[@]}"; do
  stamp=$(printf '20260726_16%02d00' "$index")
  conda run -n aligner python -m benchmarks.v4.baseline.tools.build_gt_guided_wsi_tasks \
    --cohort-manifest /nfs-medical3/zyh/v4/phase1/data/cohort200_20260716_010810/cohort_manifest.parquet \
    --class-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
    --target-class "${classes[$index]}" --split val --seed 20260726 \
    --output-root /nfs-medical3/zyh/v4/baseline --timestamp "$stamp" \
    --tile-index "${tiles[0]}" --tile-index "${tiles[1]}" --tile-index "${tiles[2]}" \
    --tile-index "${tiles[3]}" --tile-index "${tiles[4]}"
  task_dir="/nfs-medical3/zyh/v4/baseline/wsi_gt_guided_tasks_${stamp}"
  conda run -n aligner python -m benchmarks.v4.baseline.tools.build_j5_wsi_oracle_candidates \
    --task-manifest "${task_dir}/wsi_tasks_${stamp}.parquet" --candidates 5 \
    --output-root /nfs-medical3/zyh/v4/baseline --timestamp "$stamp"
done
