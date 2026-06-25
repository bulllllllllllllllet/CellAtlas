# Tissue Segmentation Benchmark

This folder evaluates CellAtlas cell tokens against color-coded 10x tissue GT.

The HE-to-GT filename rule is intentionally not hard-coded. Provide a CSV once
the matching rule is confirmed.

## Mapping CSV

Required columns:

```csv
image_id,gt_path,masks_path,reg_path,proj_path,raw_path,he_path
389502-15,/path/to/gt.png,/path/to/masks.npy,/path/to/features_reg.npy,/path/to/features_proj.npy,,/path/to/he.png
```

`raw_path` and `he_path` are optional. Use `raw_path` for CTransPath baseline if
those features are exported.

## Prepare Labels

```bash
python -m benchmarks.tissue_seg.inspect_gt_colors --gt /path/to/gt.png --top_k 32

python -m benchmarks.tissue_seg.prepare_cell_labels \
  --mapping_csv /path/to/mapping.csv \
  --output_dir /path/to/tissue_eval_prepared \
  --color_tolerance 18
```

This creates:

- `gt_masks/*_gt_mask.npy`
- `cell_labels/*_cell_labels.csv`
- `prepared_manifest.csv`

## Evaluate Clustering

```bash
python -m benchmarks.tissue_seg.eval_clustering \
  --prepared_manifest /path/to/tissue_eval_prepared/prepared_manifest.csv \
  --output_dir /path/to/results/clustering_reg \
  --head reg \
  --n_clusters 12 \
  --include_pixel_miou
```

Run the same command with `--head proj` to test the projection head.

## Evaluate Linear Probe

Optional split CSV:

```csv
image_id,split
case_a,train
case_b,test
```

Command:

```bash
python -m benchmarks.tissue_seg.eval_linear_probe \
  --prepared_manifest /path/to/tissue_eval_prepared/prepared_manifest.csv \
  --output_dir /path/to/results/linear_reg \
  --head reg \
  --split_csv /path/to/split.csv \
  --include_pixel_miou
```

Run with `--head proj` and `--head raw` to fill the full head comparison table.

## GDPH 40-Image Linear Probe Pipeline

Create a balanced 40-image split from the matched GDPH mapping:

```bash
python -m benchmarks.tissue_seg.make_linear_probe_split \
  --mapping_csv benchmarks/tissue_seg_runs/dataset_mapping.csv \
  --output_dir /nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/splits \
  --samples_per_part 8 \
  --train_per_part 6 \
  --seed 20260622
```

Build the dataset, run CellAtlas inference, prepare labels, and train both
linear probes:

```bash
conda run -n aligner python -m benchmarks.tissue_seg.run_linear_probe_pipeline \
  --selected_csv /nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/splits/selected_40.csv \
  --split_csv /nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/splits/split_30_10.csv \
  --output_root /nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1 \
  --palette_json benchmarks/tissue_seg_runs/gdph_tissue_palette.json \
  --model_path /home/zyh/NewMedLabel/CellAtlas/experiments/crc012_holdout3/20260515_160711/he_model_best.pth \
  --cuda_visible_devices 1 \
  --inference_timeout_minutes 15 \
  --run_eval \
  --include_pixel_miou
```

Summarize the two heads:

```bash
python -m benchmarks.tissue_seg.summarize_linear_probe \
  --results_root /nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/results \
  --output_csv /nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/results/summary.csv
```

The large resized HE images, masks, features, prepared labels, logs, and results
are written under `/nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1`.
Inference is run per image; images exceeding `--inference_timeout_minutes` are
skipped so one unusually slow slide does not block the whole benchmark.
