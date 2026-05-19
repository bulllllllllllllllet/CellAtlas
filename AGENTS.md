# XCellAligner — Agent Guide

## Environment & Setup

- **Python 3.10**, managed via conda env `aligner` (`environment.yml`)
- **No test framework, no linter, no type checker, no CI** — only script-based workflow
- Run everything from repo root

## Key Commands

### Data preprocessing (must run in order)

```bash
# 1. Compute mIF global normalization stats (prerequisite for training)
python compute_global_norm_stats.py --cache_dir /data/cache-mGPU

# 2. Multi-GPU feature extraction (HE → CTransPath 768-dim, mIF → CellDensityExtractor)
python pre_extract_features.py --he_dir /data/he --mif_dir /data/mIF --cache_dir /data/cache-mGPU --log_dir /data/log
```

### Training

```bash
# Single dataset
python multidata_aligner_trainer.py \
  --cache_dir /data/cache-mGPU \
  --start_index 0 \
  --mif_channel <CHANNEL_NUM> \
  --output_dir /output \
  --batch_size 64 --epochs 350 --lambda_contrast 0.01

# Multiple datasets (equal-length arrays)
python multidata_aligner_trainer.py \
  --cache_dir /data/d1/cache /data/d2/cache \
  --start_index 0 19 \
  --mif_channel 19 19 \
  --output_dir /output --batch_size 64 --epochs 350
```

### Inference

```bash
# Patch-level
python he_transformer_inference.py --image_path <path> --model_path <weights> --save_path <out> --k <clusters>

# Whole-slide (4-step pipeline: patch → stain norm → feature extraction → stitch)
python slide_inference.py --slide_path <path> --model_path <weights> --temp_path <tmp> --output_path <out> --type <organ> --k <clusters>
```

### CellEngine (programmatic API)

```python
engine = CellInferenceEngine(
    cellpose_model_path='cyto',     # or 'nuclei'
    ctranspath_checkpoint='module/checkpoint/ctranspath.pth',
    xcell_checkpoint='model.pth',
    mode='quality',                  # 'quality'=Cellpose, 'efficiency'=InstanSeg
    detail=True                      # verbose logging
)
result = engine.predict(image_source, max_cells=512)
```

## Architecture

| Layer | HE branch | mIF branch |
|---|---|---|
| Segmentation | Cellpose (quality) / InstanSeg (efficiency) | Same masks as HE (shared) |
| Feature encoder | CTransPath → 768-dim | CellDensityExtractor → N-channels |
| Global branch | ViT-Huge (`google/vit-huge-patch14-224-in21k`, frozen) | — |
| Alignment model | `XCellFormer` (small transformer + cross-attention) | — |

- Features are **padded/truncated to 255 cells** per patch (max_cells=255 in `.pkl` cache)
- mIF features are **normalized by global 99th percentile** before training (from `global_norm_stats.json`)
- CTransPath weights **must** be placed at `module/checkpoint/ctranspath.pth`
- Cache `.pkl` keys: `features` `[1, 255, D]`, `mask` `[1, 255]`, `cell_masks` `[H, W]`

## Key Files

| File | Purpose |
|---|---|
| `multidata_aligner_trainer.py` | Main training entry — contrastive + MSE loss |
| `pre_extract_features.py` | Multi-GPU feature caching (spawns `ngpu*1` workers) |
| `CellEngine.py` | Unified inference engine: segmentation → feature extraction → model inference |
| `XCellFormer.py` | Core model: small transformer for cells + optional frozen ViT-Huge for global features |
| `slide_inference.py` | Whole-slide inference pipeline |
| `he_transformer_inference.py` | Patch-level inference with KMeans clustering |
| `compute_global_norm_stats.py` | **Must run before training** — computes p99 per mIF channel |
| `loss.py` | ContrastiveLoss, InfoNCELoss, Hungarian matching loss |
| `utils.py` | `load_cellpose_model()`, `build_cell_features()` |
| `run_train.sh` | Reference script showing 4-dataset training with tuned λ values |

## Important Gotchas

- `compute_global_norm_stats.py` **must be run first** — training crashes without `global_norm_stats.json`
- CTransPath import is fragile: `from module.TransPath.ctran import ctranspath` — if the submodule is missing, inference scripts will `raise ImportError`
- Loss weighting matters: `run_train.sh` uses `--lambda_mse 100.0 --lambda_contrast 0.1` (MSE-heavy)
- mIF images are stored as multi-channel TIFFs (one file per channel, naming: `mF{id}_x{X}_y{Y}.png`)
- No `pip install -e .` — the project uses `sys.path.append` for internal imports (see `slide_inference.py`)
- `.gitignore` excludes `experiments/`, `*.pth`, `logs`, `__pycache__/`
