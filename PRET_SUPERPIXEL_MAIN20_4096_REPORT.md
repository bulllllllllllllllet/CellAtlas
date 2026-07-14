# PRET Superpixel Main20 4096 Summary

Setting: main20, HE-derived superpixels, max target dimension 4096, realistic/scribble prompts are positive-only; oracle_gt_purity may use GT-filtered negatives.

## realistic_box / exclude prompt region

| Variant | mAP | AUROC | P@top5% area | top10 Dice | Prompt target frac |
|---|---:|---:|---:|---:|---:|
| image_cell_reg_cellw0p5 | 0.6693 | 0.8677 | 0.6701 | 0.4262 | 0.952 |
| image_cell_reg_cellw0p25 | 0.6676 | 0.8672 | 0.6663 | 0.4274 | 0.952 |
| image_cell_reg | 0.6523 | 0.8590 | 0.6621 | 0.4129 | 0.952 |
| image_cell_reg_cellw2p0 | 0.6315 | 0.8481 | 0.6478 | 0.4006 | 0.952 |
| image_only | 0.6179 | 0.8520 | 0.6182 | 0.4029 | 0.952 |
| cell_reg | 0.6156 | 0.8408 | 0.6343 | 0.3895 | 0.952 |
| image_patch_cell_reg | 0.6026 | 0.8289 | 0.6275 | 0.3926 | 0.952 |
| patch_only | 0.4648 | 0.7303 | 0.5034 | 0.2986 | 0.952 |
| random_token | 0.2140 | 0.4962 | 0.2047 | 0.1082 | 0.952 |

## scribble_like / exclude prompt region

| Variant | mAP | AUROC | P@top5% area | top10 Dice | Prompt target frac |
|---|---:|---:|---:|---:|---:|
| image_cell_reg_cellw0p5 | 0.6834 | 0.8741 | 0.6874 | 0.4428 | 0.981 |
| image_cell_reg_cellw0p25 | 0.6810 | 0.8734 | 0.6849 | 0.4425 | 0.981 |
| image_cell_reg | 0.6644 | 0.8647 | 0.6767 | 0.4277 | 0.981 |
| image_cell_reg_cellw2p0 | 0.6421 | 0.8535 | 0.6609 | 0.4139 | 0.981 |
| image_only | 0.6295 | 0.8567 | 0.6347 | 0.4149 | 0.981 |
| cell_reg | 0.6249 | 0.8460 | 0.6443 | 0.4016 | 0.981 |
| image_patch_cell_reg | 0.6133 | 0.8334 | 0.6365 | 0.4019 | 0.981 |
| patch_only | 0.4713 | 0.7311 | 0.5098 | 0.3030 | 0.981 |
| random_token | 0.2148 | 0.4962 | 0.2069 | 0.1089 | 0.981 |

## oracle_gt_purity / exclude prompt region

| Variant | mAP | AUROC | P@top5% area | top10 Dice | Prompt target frac |
|---|---:|---:|---:|---:|---:|
| image_cell_reg_cellw0p5 | 0.7072 | 0.8804 | 0.7014 | 0.4719 | 1.000 |
| image_cell_reg | 0.7015 | 0.8817 | 0.7072 | 0.4644 | 1.000 |
| image_cell_reg_cellw0p25 | 0.6887 | 0.8716 | 0.6743 | 0.4621 | 1.000 |
| image_cell_reg_cellw2p0 | 0.6853 | 0.8766 | 0.6997 | 0.4533 | 1.000 |
| cell_reg | 0.6706 | 0.8711 | 0.6888 | 0.4427 | 1.000 |
| image_patch_cell_reg | 0.6658 | 0.8654 | 0.6897 | 0.4446 | 1.000 |
| image_only | 0.6361 | 0.8487 | 0.6130 | 0.4270 | 1.000 |
| patch_only | 0.5112 | 0.7554 | 0.5502 | 0.3321 | 1.000 |
| random_token | 0.2276 | 0.4987 | 0.2188 | 0.1146 | 1.000 |

## Per-class realistic_box / exclude prompt region

### image_only

| Class | Queries | mAP | P@top5% area | Dice | Low-cell focus |
|---:|---:|---:|---:|---:|---|
| 0 | 288 | 0.863 | 0.905 | 0.468 | False |
| 1 | 312 | 0.460 | 0.517 | 0.302 | False |
| 3 | 252 | 0.235 | 0.218 | 0.210 | True |
| 4 | 102 | 0.678 | 0.659 | 0.499 | False |
| 6 | 207 | 0.752 | 0.695 | 0.571 | False |
| 7 | 360 | 0.792 | 0.835 | 0.502 | False |
| 8 | 93 | 0.495 | 0.274 | 0.287 | False |
| 9 | 36 | 0.355 | 0.258 | 0.238 | True |
| 10 | 3 | 0.015 | 0.000 | 0.000 | True |
| 11 | 3 | 1.000 | 0.045 | 0.054 | False |

### cell_reg

| Class | Queries | mAP | P@top5% area | Dice | Low-cell focus |
|---:|---:|---:|---:|---:|---|
| 0 | 288 | 0.881 | 0.927 | 0.454 | False |
| 1 | 312 | 0.478 | 0.571 | 0.305 | False |
| 3 | 252 | 0.408 | 0.385 | 0.313 | True |
| 4 | 102 | 0.540 | 0.560 | 0.407 | False |
| 6 | 207 | 0.495 | 0.522 | 0.388 | False |
| 7 | 360 | 0.760 | 0.820 | 0.470 | False |
| 8 | 93 | 0.734 | 0.408 | 0.445 | False |
| 9 | 36 | 0.308 | 0.267 | 0.214 | True |
| 10 | 3 | 0.018 | 0.000 | 0.000 | True |
| 11 | 3 | 1.000 | 0.045 | 0.052 | False |

### image_cell_reg_cellw0p25

| Class | Queries | mAP | P@top5% area | Dice | Low-cell focus |
|---:|---:|---:|---:|---:|---|
| 0 | 288 | 0.895 | 0.934 | 0.474 | False |
| 1 | 312 | 0.498 | 0.564 | 0.325 | False |
| 3 | 252 | 0.391 | 0.358 | 0.277 | True |
| 4 | 102 | 0.704 | 0.664 | 0.522 | False |
| 6 | 207 | 0.733 | 0.701 | 0.563 | False |
| 7 | 360 | 0.804 | 0.854 | 0.504 | False |
| 8 | 93 | 0.694 | 0.396 | 0.424 | False |
| 9 | 36 | 0.367 | 0.303 | 0.258 | True |
| 10 | 3 | 0.017 | 0.000 | 0.000 | True |
| 11 | 3 | 1.000 | 0.045 | 0.055 | False |

### image_cell_reg_cellw0p5

| Class | Queries | mAP | P@top5% area | Dice | Low-cell focus |
|---:|---:|---:|---:|---:|---|
| 0 | 288 | 0.905 | 0.940 | 0.472 | False |
| 1 | 312 | 0.510 | 0.582 | 0.331 | False |
| 3 | 252 | 0.425 | 0.396 | 0.304 | True |
| 4 | 102 | 0.708 | 0.653 | 0.508 | False |
| 6 | 207 | 0.679 | 0.668 | 0.518 | False |
| 7 | 360 | 0.796 | 0.844 | 0.501 | False |
| 8 | 93 | 0.719 | 0.404 | 0.448 | False |
| 9 | 36 | 0.340 | 0.308 | 0.247 | True |
| 10 | 3 | 0.020 | 0.000 | 0.000 | True |
| 11 | 3 | 1.000 | 0.046 | 0.053 | False |

### random_token

| Class | Queries | mAP | P@top5% area | Dice | Low-cell focus |
|---:|---:|---:|---:|---:|---|
| 0 | 288 | 0.363 | 0.346 | 0.149 | False |
| 1 | 312 | 0.194 | 0.185 | 0.111 | False |
| 3 | 252 | 0.068 | 0.060 | 0.065 | True |
| 4 | 102 | 0.111 | 0.105 | 0.087 | False |
| 6 | 207 | 0.136 | 0.127 | 0.090 | False |
| 7 | 360 | 0.354 | 0.350 | 0.145 | False |
| 8 | 93 | 0.035 | 0.023 | 0.038 | False |
| 9 | 36 | 0.050 | 0.043 | 0.053 | True |
| 10 | 3 | 0.002 | 0.000 | 0.000 | True |
| 11 | 3 | 0.003 | 0.000 | 0.000 | False |

## Initial interpretation

- The best overall realistic positive-only result is image_cell_reg_cellw0p5: mAP 0.6693, AUROC 0.8677, top10 Dice 0.4262.
- Cell features help when weighted moderately. cellw0p25/0p5 outperform image_only and the equal-weight image_cell_reg; cellw2p0 degrades, so too much cell weight adds noise.
- random_token is much lower, confirming the retrieval is not just class prevalence or prompt leakage.
- patch_only is weak at this 4096 superpixel setting, and image_patch_cell_reg does not improve, suggesting 1024 patch context is too coarse or underweighted for this prompt task.
- realistic_box prompt target fraction is about 0.952, so the automatic query boxes are still fairly clean; this should be stated when reporting.
