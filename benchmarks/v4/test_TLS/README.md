# Held-out TLS prompt retrieval

This experiment parses KFB `Curve` polygons in level-0 coordinates, uses a
subset as positive prompts, chooses negative prompts from non-annotated tissue,
and evaluates retrieval on both prompted and held-out TLS polygons.

The HE-only cell path reuses `new_inference_stream`: Cellpose segmentation,
soft-attention CTransPath-768, and XCellFormer reg-64. Because external KFB
slides do not provide GDPH nuclei classes, class ID 0 is recorded explicitly
as unknown; this is an audited domain difference, not a generated class label.

## Tested case

- KFB dimensions: `76045 x 47283` at level 0; inference canvas
  `19011 x 11820` at 10x.
- Six `Curve` polygons are parsed from the annotation JSON.
- Polygon indices `0, 4, 5` provide positive prompts; indices `1, 2, 3` are
  held out from prompting and measure retrieval.
- Three tissue points outside dilated TLS annotations provide negative prompts.

## Pipeline

Run from the repository root with the `aligner` conda environment:

```bash
conda run -n aligner python benchmarks/v4/test_TLS/prepare_tls_case.py \
  --wsi-path "/path/to/case.kfb" \
  --annotation-json "/path/to/Annotations/1.json"

conda run -n aligner python benchmarks/v4/test_TLS/extract_tls_cell_features.py \
  --tile-index /path/to/tls_case_TIMESTAMP/wsi_tile_index.parquet \
  --device cuda:5 --xcell-batch-size 8

conda run -n aligner python -m benchmarks.v4.whole_slide_inference.infer_wsi \
  --config benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_j5_full_budget.yaml \
  --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
  --phase5-config benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml \
  --phase2-checkpoint /path/to/phase2.pth \
  --cell-checkpoint /path/to/phase3.pth \
  --phase5-checkpoint /path/to/phase5.pth \
  --joint-checkpoint /path/to/j5.pth \
  --tile-index /path/to/tls_case_TIMESTAMP/wsi_tile_index.parquet \
  --cell-feature-manifest /path/to/tls_cell_features_TIMESTAMP/feature_index.parquet \
  --prompt-json /path/to/tls_case_TIMESTAMP/prompts.json \
  --gpus 5 --batch-size 2 --num-workers 4

conda run -n aligner python benchmarks/v4/test_TLS/evaluate_tls_retrieval.py \
  --inference-dir /path/to/wsi_inference_TIMESTAMP \
  --case-manifest /path/to/tls_case_TIMESTAMP/case_manifest.json \
  --annotation-json "/path/to/Annotations/1.json"
```

The evaluation writes union and per-polygon metrics plus a coordinate-matched
four-panel review image. Per-polygon recall is meaningful for retrieval.
Per-polygon precision includes predictions on the other five TLS polygons as
false positives; use the union precision/Dice for overall segmentation quality.
