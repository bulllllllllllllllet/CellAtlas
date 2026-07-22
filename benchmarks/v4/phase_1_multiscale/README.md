# Phase 1 10x supervised baseline

This directory implements only the data contract and supervised 10x baseline
that precede the learned region, cell, prompt, and mask-decoder modules.

All commands are run from repository root with the `aligner` environment. Each
command creates a new timestamped artifact directory under
`/nfs-medical3/zyh/v4`; no source image or pair CSV is edited.

1. Audit the explicit pairs and inspect the generated overlay previews:

```bash
conda run -n aligner python benchmarks/v4/phase_1_multiscale/tools/audit_10x_dataset.py \
  --config benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml
```

2. Audit confirms the configured exact GDPH 12-class palette. Unknown or
unannotated colors remain `ignore_index=255`; any mismatch is a data-audit
failure, not a nearest-color conversion.

3. Build the manifest, then use its path to create the dynamic patch index:

```bash
conda run -n aligner python benchmarks/v4/phase_1_multiscale/tools/build_wsi_gt_manifest.py \
  --config benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml

conda run -n aligner python benchmarks/v4/phase_1_multiscale/tools/build_10x_patch_index.py \
  --config benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml \
  --manifest /nfs-medical3/zyh/v4/phase1/data/manifest_<timestamp>/wsi_gt_manifest.parquet
```

4. Run the tests and the required 16-patch overfit gate. `pretrained_weights`
must point to a local compatible checkpoint. Download the official public
COCO DeepLabV3-ResNet50 checkpoint first if needed:

```bash
conda run -n aligner python benchmarks/v4/phase_1_multiscale/tools/download_pretrained_deeplabv3.py \
  --config benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml
```

Set `model.pretrained_weights` to the printed checkpoint path. The 21-class
COCO prediction head is intentionally replaced by the trainable 12-class head.

```bash
conda run -n aligner python -m unittest discover -s benchmarks/v4/phase_1_multiscale/tests -v

conda run -n aligner python benchmarks/v4/phase_1_multiscale/train_phase1_10x.py \
  --config benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml \
  --patch-index /nfs-medical3/zyh/v4/phase1/data/patch_index_<timestamp>/patch_index_10x.parquet \
  --overfit-num-patches 16
```

Only after reviewing alignment, tests, overfit output, and a one-epoch smoke
test should a full DDP training job be started.
