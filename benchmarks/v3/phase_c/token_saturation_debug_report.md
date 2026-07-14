# Token / Score Saturation Debug Report

## Trigger

The first Phase E visualization (`1512244-15-HE-DX1_c0_msp_noisy_001`, small scale) showed an almost entirely red Phase C score map.

## Verified Facts

- The saved `score_final` exactly matches recomputation from the enhanced token (`max_abs_error = 0`). There is no score/token file mismatch.
- This positive-only query is genuinely saturated: mean cosine 0.9860, median 0.9955, std 0.0266, and 86.0% of valid superpixels score at least 0.98.
- The heat renderer already uses p02-p98 scaling, but the score distribution is strongly skewed: p02=0.8939, median=0.9955, p98=0.9996. Therefore most regions still render red. The Phase E visualizer now shows `rank_percentile` and writes raw score quantiles in the caption.
- Pairwise superpixel cosine on this image is high for all raw statistical token variants: image-only 0.9920, base cellw0.5 0.9682, enhanced 0.9743. This is a common-direction / anisotropy issue in the region token representation.
- The underlying per-cell `reg.npy` is not collapsed: shape 675353x64, all finite, no constant dimensions, mean dimension std 0.1496, effective rank 10.39. Cell-to-superpixel counts are also plausible (656014 assigned cells; 6.89% empty small superpixels).
- All 20 cell caches pass feature finiteness, count alignment, and label validation (12,880,537 total cells).

## Controls

For the suspicious query:

| token | score std | mAP | AUROC |
|---|---:|---:|---:|
| image-only | 0.0058 | 0.7508 | 0.8900 |
| cell-only | 0.2469 | 0.6844 | 0.8628 |
| base cellw0.5 | 0.0304 | 0.7117 | 0.8768 |
| base cellw2.0 | 0.1380 | 0.6884 | 0.8645 |
| enhanced | 0.0266 | 0.7169 | 0.8836 |
| enhanced, per-image mean removed | 0.4577 | 0.7413 | 0.8974 |

The color-only token is better on this single query but is even more cosine-saturated.

On a stratified 300-query control, original enhanced beats original image-only for every prompt quality:

| quality | enhanced mAP/AUROC | image-only mAP/AUROC |
|---|---:|---:|
| clean | 0.6146 / 0.8946 | 0.5885 / 0.8859 |
| noisy | 0.4399 / 0.8605 | 0.4149 / 0.8517 |
| hard negative | 0.6119 / 0.8915 | 0.5246 / 0.8584 |

Per-image mean removal expands the score range but slightly hurts clean and hard-negative retrieval; it only improves noisy prompts. It is not a safe formal replacement.

## Decision

- Do **not** rerun cell feature extraction. The source cell-reg arrays and cell/superpixel alignment validations do not show corruption.
- Do **not** overwrite formal tokens with simple centering. The cross-query control does not support it.
- Treat cosine saturation as a token geometry/design limitation, not a broken cache. Future token work should be a separate ablation using training-fold centering/whitening or a learned metric projection from the existing cell features.
- Continue to use enhanced token as the current formal input because it remains stronger than image-only and the previous five-variant token ablation winner.
