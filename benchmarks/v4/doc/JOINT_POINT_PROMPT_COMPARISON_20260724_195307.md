# Joint model point-prompt comparison — 2026-07-24

## Scope

Validation split only; no test set was used. All prompt coordinates are frozen from
Phase-2 hard-assignment member pixels and do not use GT to choose prompt locations.
The target is the Phase-6 J5 checkpoint:

`/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth`

The point-only protocols use the first frozen positive points and first frozen
negative point of each eligible occurrence. No bounding-box experiment is reported:
the joint prompt encoder has no bbox input branch and has not been trained with bbox
prompts, so such a result would not be a valid comparison.

## Results

All metrics are pixel-level. Class-macro Dice and mIoU are macro-averaged over the
12 target classes. Boundary F1 uses a 2-pixel tolerance.

| Prompt protocol | Method | Episodes | Global Dice | Class-macro Dice | mIoU | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| 1 positive + 1 negative | SAM | 4000 | 0.4860 | 0.4649 | 0.3094 | 0.1369 |
| 1 positive + 1 negative | SAM-Med2D | 4000 | 0.2119 | 0.2318 | 0.1379 | 0.0954 |
| 1 positive + 1 negative | WSI-SAM | 4000 | 0.4154 | 0.4064 | 0.2653 | 0.1353 |
| 1 positive + 1 negative | Joint model J5 | 4000 | **0.7522** | **0.7418** | **0.5938** | **0.2971** |
| 3 positive + 1 negative | Joint model J5 | 1261 | 0.8164 | 0.8159 | 0.6913 | 0.3324 |
| 5 positive + 1 negative | Joint model J5 | 885 | 0.8225 | 0.8206 | 0.6978 | 0.3229 |

Under the identical 1-positive/1-negative 4000-episode protocol, the joint model
achieves 0.7522 global Dice, versus 0.4860 for SAM, 0.4154 for WSI-SAM, and 0.2119
for SAM-Med2D.

## Interpretation limits

- The 3+1 and 5+1 rows do **not** use the same occurrence cohort as the 1+1 row.
  The frozen source contains enough positive points for only 1261 and 885 episodes,
  respectively. They show performance under additional points, but are not a strict
  causal point-count curve against the 4000-episode 1+1 result.
- A strict 1/3/5 comparison requires rerunning all three budgets on their common
  885-episode eligible subset.
- SAM, SAM-Med2D, and WSI-SAM have been completed only for the 1+1 protocol;
  their 3+1 and 5+1 rows are intentionally absent rather than inferred.

## Frozen manifests

| Protocol | Manifest | Episodes |
|---|---|---:|
| 1+1 | `/nfs-medical3/zyh/v4/baseline/point_1p_1n_manifest_20260724_173500/episode_manifest_20260724_173500.parquet` | 4000 |
| 3+1 | `/nfs-medical3/zyh/v4/baseline/point_3p_1n_manifest_20260724_173501/episode_manifest_20260724_173501.parquet` | 1261 |
| 5+1 | `/nfs-medical3/zyh/v4/baseline/point_5p_1n_manifest_20260724_173502/episode_manifest_20260724_173502.parquet` | 885 |

## Raw outputs

### Joint model J5

- 1+1: `/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260724_173800`
- 3+1: `/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260724_173801`
- 5+1: `/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260724_173802`
- Sequential execution log: `/nfs-medical3/zyh/v4/phase6/evaluation/joint_point_curve_20260724_173800.log`

### External baselines, 1+1 only

- SAM: `/nfs-medical3/zyh/v4/baseline/sam_20260724_000200`
- SAM-Med2D: `/nfs-medical3/zyh/v4/baseline/sam_med2d_20260724_000201`
- WSI-SAM: `/nfs-medical3/zyh/v4/baseline/wsi_sam_20260724_000202`

## SAM-Med2D input audit

The adapter was compared line-by-line with the pinned upstream implementation:

- RGB input agrees with the upstream model's declared `image_format`.
- ImageNet normalization, 256x256 nearest-neighbour resize, point/box coordinate
  scaling, predicted-IoU candidate selection, and bilinear mask upsampling agree
  with the upstream `SammedPredictor` path.
- A direct numerical invocation of `SammedPredictor` is not currently available in
  the `aligner` environment because its optional upstream dependency
  `albumentations` is absent. This is an unclosed audit dependency, not evidence of
  an identified input-mapping error; the existing SAM-Med2D score has not been
  silently altered or rerun.
