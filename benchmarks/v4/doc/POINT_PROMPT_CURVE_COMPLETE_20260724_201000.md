# Complete point-prompt comparison — 2026-07-24

## Scope

Validation split only; no test set was used. Each row uses frozen Phase-2
hard-assignment prompt coordinates with no GT-based prompt selection. Point budgets
are `N` positive points plus one negative point, with no bbox.

The joint-model rows use the J5 checkpoint:

`/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth`

The joint prompt encoder has no bbox input branch and was not trained on bbox prompts,
so bbox is intentionally excluded rather than treated as an invalid model comparison.

## Complete results

All metrics are pixel-level. Each episode is a binary target-class-versus-rest
segmentation task. For each of the 12 target classes, TP, FP, and FN are pooled
over that class's episodes before computing foreground Dice and foreground IoU.
The class-macro metrics are the unweighted means of those 12 class-level values.
In particular, foreground IoU is `TP / (TP + FP + FN)`: TN and background IoU
are not included. Boundary F1 uses a 2-pixel tolerance. Every external-baseline
run completed all assigned episodes with zero failures.

| Prompt protocol | Episodes | Method | Global foreground Dice | Class-macro foreground Dice | Class-macro foreground IoU | Boundary F1 |
|---|---:|---|---:|---:|---:|---:|
| 1 positive + 1 negative | 4000 | SAM | 0.4860 | 0.4649 | 0.3094 | 0.1369 |
| 1 positive + 1 negative | 4000 | SAM-Med2D | 0.2119 | 0.2318 | 0.1379 | 0.0954 |
| 1 positive + 1 negative | 4000 | WSI-SAM | 0.4154 | 0.4064 | 0.2653 | 0.1353 |
| 1 positive + 1 negative | 4000 | Joint model J5 | **0.7522** | **0.7418** | **0.5938** | **0.2971** |
| 3 positive + 1 negative | 1261 | SAM | 0.5903 | 0.5805 | 0.4143 | 0.1763 |
| 3 positive + 1 negative | 1261 | SAM-Med2D | 0.3178 | 0.3323 | 0.2089 | 0.0980 |
| 3 positive + 1 negative | 1261 | WSI-SAM | 0.5623 | 0.5608 | 0.3976 | 0.1676 |
| 3 positive + 1 negative | 1261 | Joint model J5 | **0.8164** | **0.8159** | **0.6913** | **0.3324** |
| 5 positive + 1 negative | 885 | SAM | 0.6290 | 0.6243 | 0.4585 | 0.1956 |
| 5 positive + 1 negative | 885 | SAM-Med2D | 0.4050 | 0.4149 | 0.2706 | 0.1091 |
| 5 positive + 1 negative | 885 | WSI-SAM | 0.6041 | 0.6057 | 0.4397 | 0.1764 |
| 5 positive + 1 negative | 885 | Joint model J5 | **0.8225** | **0.8206** | **0.6978** | **0.3229** |

## Readout

- On the identical 1+1, 4000-episode protocol, J5 global Dice is 0.7522, compared
  with SAM 0.4860, WSI-SAM 0.4154, and SAM-Med2D 0.2119.
- On the 3+1 and 5+1 cohorts, J5 remains the leading method on every reported
  metric. SAM is the strongest external baseline in those two budgets.
- Point-count rows use progressively smaller eligible cohorts: 4000 (1+1), 1261
  (3+1), and 885 (5+1). Thus, each row is a valid cross-method comparison within
  its own prompt protocol, but raw performance changes across rows are not a strict
  causal point-count curve. A strict 1/3/5 curve requires all budgets to be rerun
  on the common 885-episode cohort.
- Global foreground Dice is pixel-pooled and therefore gives more weight to
  classes with more foreground pixels. The two class-macro foreground metrics
  deliberately give each target class equal weight, so a rare or difficult class
  can lower them substantially. Report both views: the global score summarizes
  overall foreground-pixel performance, while the class-macro scores expose
  failures that would otherwise be hidden by common or large classes.

## Frozen manifests

| Protocol | Manifest |
|---|---|
| 1+1 | `/nfs-medical3/zyh/v4/baseline/point_1p_1n_manifest_20260724_173500/episode_manifest_20260724_173500.parquet` |
| 3+1 | `/nfs-medical3/zyh/v4/baseline/point_3p_1n_manifest_20260724_173501/episode_manifest_20260724_173501.parquet` |
| 5+1 | `/nfs-medical3/zyh/v4/baseline/point_5p_1n_manifest_20260724_173502/episode_manifest_20260724_173502.parquet` |

## Raw result directories

| Method | 1+1 | 3+1 | 5+1 |
|---|---|---|---|
| SAM | `/nfs-medical3/zyh/v4/baseline/sam_20260724_000200` | `/nfs-medical3/zyh/v4/baseline/sam_20260724_200000` | `/nfs-medical3/zyh/v4/baseline/sam_20260724_200001` |
| SAM-Med2D | `/nfs-medical3/zyh/v4/baseline/sam_med2d_20260724_000201` | `/nfs-medical3/zyh/v4/baseline/sam_med2d_20260724_200010` | `/nfs-medical3/zyh/v4/baseline/sam_med2d_20260724_200011` |
| WSI-SAM | `/nfs-medical3/zyh/v4/baseline/wsi_sam_20260724_000202` | `/nfs-medical3/zyh/v4/baseline/wsi_sam_20260724_200020` | `/nfs-medical3/zyh/v4/baseline/wsi_sam_20260724_200021` |
| Joint model J5 | `/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260724_173800` | `/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260724_173801` | `/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260724_173802` |

## SAM-Med2D audit status

The adapter's RGB input, normalization, 256x256 nearest-neighbour resize, prompt
coordinate scaling, candidate selection, and mask upsampling were checked against
the pinned upstream source. Direct numerical replay through the upstream
`SammedPredictor` remains unavailable in the `aligner` environment because
`albumentations` is absent. This dependency remains an explicitly unclosed audit
item; no score was silently changed or discarded.
