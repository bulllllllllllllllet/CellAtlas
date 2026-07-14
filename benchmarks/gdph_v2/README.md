# GDPH External Validation v2

All outputs are written under:

`/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2`

The experiment uses the original level-0 HE image for Cellpose and CellAtlas.
Cell centers and polygons are mapped to the 10x tissue GT only after inference.
Full-resolution inference uses 4096-pixel core tiles with a 256-pixel halo and
keeps cells whose centroids belong to the non-overlapping core. Each tile is
atomically cached, so an interrupted slide resumes from the first missing tile.

## Required order

Run from the CellAtlas repository root.

```bash
python -m benchmarks.gdph_v2.init_experiment
python -m benchmarks.gdph_v2.select_pilot
python -m benchmarks.gdph_v2.validate_setup

CUDA_VISIBLE_DEVICES=1 conda run -n aligner \
  python -m benchmarks.gdph_v2.gpu_preflight
CUDA_VISIBLE_DEVICES=1 conda run -n aligner \
  python -u -m benchmarks.gdph_v2.fullres_inference

python -m benchmarks.gdph_v2.polygon_labels
conda run -n aligner python -m benchmarks.gdph_v2.eval_nucleus_detection

python -m benchmarks.gdph_v2.select_main
CUDA_VISIBLE_DEVICES=1 conda run -n aligner \
  python -u -m benchmarks.gdph_v2.fullres_inference \
  --manifest /nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/manifests/main_20.csv
python -m benchmarks.gdph_v2.polygon_labels \
  --manifest /nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/manifests/main_20.csv
conda run -n aligner python -m benchmarks.gdph_v2.eval_crossval

python -m benchmarks.gdph_v2.generate_queries
python -m benchmarks.gdph_v2.eval_retrieval
conda run -n aligner python -m benchmarks.gdph_v2.eval_cell_coverage

CUDA_VISIBLE_DEVICES=1 conda run -n aligner \
  python -u -m benchmarks.gdph_v2.patch_inference
conda run -n aligner python -m benchmarks.gdph_v2.eval_patch_retrieval
python -m benchmarks.gdph_v2.audit_experiment
```

Do not start a long GPU stage unless `gpu_preflight.json` has `passed: true`.
The raw, regression, projection, cell metadata, and polygon counts must be
strictly equal. No evaluator silently truncates mismatched arrays.

## PRET-style superpixel in-context benchmark

This optional stage evaluates prompt-driven tissue retrieval/segmentation on
10x HE superpixels. Superpixels are generated from HE only; GT is used only for
prompt simulation and metrics.

```bash
conda run -n aligner python -m benchmarks.gdph_v2.pret_superpixel_tokens \
  --manifest /nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/manifests/main_20.csv \
  --output_root /nfs-medical3/zyh/cellatlas_gdph_benchmark_v2 \
  --workers 8

conda run -n aligner python -m benchmarks.gdph_v2.pret_generate_prompts \
  --output_root /nfs-medical3/zyh/cellatlas_gdph_benchmark_v2

conda run -n aligner python -m benchmarks.gdph_v2.pret_eval_in_context \
  --output_root /nfs-medical3/zyh/cellatlas_gdph_benchmark_v2 \
  --workers 8

conda run -n aligner python -m benchmarks.gdph_v2.pret_visualize \
  --output_root /nfs-medical3/zyh/cellatlas_gdph_benchmark_v2
```

Outputs are written under `pret_superpixel/`. Keep `oracle_gt_purity`,
`realistic_box`, and `scribble_like` prompt summaries separate in reports.
