# Baseline single-patch metrics

## Scope

These are `limit=1` validation gates, not cohort-level results. All three
methods use the same 512 x 512 10x patch (`episode_index=19522`,
`1134023-3-HE-DX1__x10240_y11264`), target class `muscle` (class 7), five
frozen positive points, three frozen negative points, and the frozen positive
box because this is a `large` prompt episode. Dice is calculated against the
GT muscle-vs-rest mask within this patch.

| Method | Status | Pixel Dice | Boundary F1 (2 px) | TP | FP | FN | TN | Latency (ms) | Peak GPU memory (MiB) | Candidate rule |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SAM (ViT-B) | completed | 0.4237 | 0.1218 | 42,157 | 461 | 114,211 | 105,315 | 1105.6 | 2772.2 | Max predicted IoU, 3 candidates |
| SAM-Med2D | completed | 0.4063 | 0.1062 | 40,077 | 848 | 116,291 | 104,928 | 549.4 | 1121.5 | Max predicted IoU, 3 candidates |
| WSI-SAM | completed | 0.3904 | 0.0851 | 38,234 | 1,245 | 118,134 | 104,531 | 591.4 | 342.0 | Single fused mask |

## Metric definitions

| Metric | Definition |
|---|---|
| TP / FP / FN / TN | Pixel counts against the target-class binary GT inside valid pixels. |
| Pixel Dice | `2*TP / (2*TP + FP + FN)`. Higher is better. |
| Boundary F1 (2 px) | F1 score between prediction and target boundaries, allowing 2-pixel tolerance. Higher is better. |
| Latency | Wall-clock model inference time for this single patch. |
| Peak GPU memory | Peak CUDA memory reported by the evaluation adapter. |

## Evidence artifacts

- SAM: `/nfs-medical3/zyh/v4/baseline/sam_20260723_201600/`
- SAM-Med2D: `/nfs-medical3/zyh/v4/baseline/sam_med2d_20260723_201601/`
- WSI-SAM: `/nfs-medical3/zyh/v4/baseline/wsi_sam_20260723_201602/`

Do not interpret this table as a full validation ranking. It contains one
large-prompt muscle patch only; full-manifest evaluation is required for
cohort-level Dice, latency, and failure-rate comparisons.
