# CellAtlas v4 whole-slide inference

This package applies the formal J5 prompt-conditioned model to every tile of
one WSI and stitches the probabilities in 10x coordinates. It does not use the
legacy root-level WSI pipeline and it does not enable the rejected Phase4
parent-context branch.

## Input contract

J5 requires the same cell branch used during training. Whole-slide inference
therefore has three explicit preparation steps:

1. Build a complete overlapping tile index.
2. Run the existing Phase3 XCell extractor on that exact index. The source WSI
   must have aligned nuclei instance/class maps in the Phase1 pairs config.
3. Supply positive and negative prompts in level-0 WSI pixel coordinates.

Example tile-index command:

```bash
conda run -n aligner python -m benchmarks.v4.whole_slide_inference.build_wsi_tile_index \
  --wsi-path /path/to/slide.tiff --wsi-id SLIDE_ID \
  --level0-downsample 2 --tile-size 512 --stride 384 \
  --output-root /nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes
```

Use the resulting `wsi_tile_index.parquet` as `--patch-index` for
`phase_3_cell_region/tools/extract_xcell_features.py`. Its generated
`feature_index.parquet` is the inference command's
`--cell-feature-manifest`.

Prompt JSON schema:

```json
{
  "coordinate_space": "level0",
  "prompt_size": "point",
  "positive": [{"x": 12000, "y": 8000}],
  "negative": [{"x": 16000, "y": 8000}]
}
```

Run inference with the same J5/upstream config and checkpoint arguments used
by Phase6 evaluation:

```bash
conda run -n aligner python -m benchmarks.v4.whole_slide_inference.infer_wsi \
  --config benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_j5_full_budget.yaml \
  --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
  --phase5-config benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml \
  --phase2-checkpoint /path/to/phase2.pth \
  --cell-checkpoint /path/to/phase3.pth \
  --phase5-checkpoint /path/to/phase5.pth \
  --joint-checkpoint /path/to/j5_epoch4.pth \
  --tile-index /path/to/wsi_tile_index.parquet \
  --cell-feature-manifest /path/to/feature_index.parquet \
  --prompt-json /path/to/prompts.json \
  --gpus 0 --batch-size 2 --num-workers 4
```

The timestamped output contains tiled pyramidal TIFFs for probability, binary
mask, and blend coverage; disk-backed accumulators; `completed.jsonl` and
`failures.jsonl`; and complete provenance in `metadata.json`.

Positive and negative clicks are resolved to learned regions in their source
tiles. Those region tokens are encoded once into a global task token. The same
task token is then applied to all WSI tiles. If a positive and a negative click
resolve to the same learned region, the run records `abstained_prompt_conflict`
and does not emit a mask.
