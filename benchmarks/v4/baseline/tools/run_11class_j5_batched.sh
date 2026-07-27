#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaoyh/CellAtlas
classes=(tumor_epithelium tumor_stroma background necrosis normal_gland normal_stroma submucosa_serosa lymphocyte_aggregate mucus fat blood)
wsis=(1033746-12-HE-DX1 1028417-R1-HE-DX1 1321593-10-HE-DX1 1416664-10-HE-DX1 1504774-12-HE-DX1)
tiles=(
 /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1033746-12-HE-DX1_20260725_194443/wsi_tile_index.parquet
 /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1028417-R1-HE-DX1_20260725_200000/wsi_tile_index.parquet
 /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1321593-10-HE-DX1_20260725_200000/wsi_tile_index.parquet
 /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1416664-10-HE-DX1_20260725_200000/wsi_tile_index.parquet
 /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes/wsi_tile_index_1504774-12-HE-DX1_20260725_200000/wsi_tile_index.parquet
)
cells=(
 /nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_113000/feature_index.parquet
 /nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_113001/feature_index.parquet
 /nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_113002/feature_index.parquet
 /nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_114917/feature_index.parquet
 /nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_114918/feature_index.parquet
)
for ci in "${!classes[@]}"; do
  task_stamp=$(printf '20260726_16%02d00' "$ci")
  candidates="/nfs-medical3/zyh/v4/baseline/j5_wsi_oracle_candidates_${task_stamp}/candidate_manifest_${task_stamp}.parquet"
  for wi in "${!wsis[@]}"; do
    if (( ci == 0 && wi == 0 )); then
      continue
    fi
    run_stamp=$(printf '20260726_18%02d%02d' "$ci" "$wi")
    conda run -n aligner python -m benchmarks.v4.whole_slide_inference.infer_wsi_multi_prompt \
      --config benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_j5_full_budget.yaml \
      --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
      --phase5-config benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml \
      --phase2-checkpoint /nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth \
      --cell-checkpoint /nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth \
      --phase5-checkpoint /nfs-medical3/zyh/v4/phase5/runs/phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth \
      --joint-checkpoint /nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth \
      --tile-index "${tiles[$wi]}" --cell-feature-manifest "${cells[$wi]}" \
      --candidate-manifest "$candidates" --wsi-id "${wsis[$wi]}" --gpu 5 \
      --batch-size 2 --num-workers 4 --output-root /nfs-medical3/zyh/v4/whole_slide_inference \
      --timestamp "$run_stamp"
  done
done
