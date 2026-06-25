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

