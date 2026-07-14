# Token Geometry Ablation

Strict 12/4/4 WSI-level direct score calibration using classwise top-area fraction.

| candidate | Pixel Dice | mAP | AUROC | score std | frac(score>0.98) |
|---|---:|---:|---:|---:|---:|
| cell | 0.3885 | 0.4655 | 0.7834 | 0.2189 | 0.0741 |
| cellstats | 0.3490 | 0.3447 | 0.7756 | 0.0193 | 0.4572 |
| enhanced_centered | 0.4489 | 0.5651 | 0.8736 | 0.3515 | 0.0229 |
| enhanced_raw | 0.4500 | 0.5692 | 0.8809 | 0.0311 | 0.2853 |
| enhanced_raw_logsumexp | 0.4538 | 0.5807 | 0.8838 | 0.0323 | 0.2010 |
| enhanced_raw_mean | 0.4496 | 0.5759 | 0.8825 | 0.0325 | 0.1954 |
| enhanced_raw_median | 0.4428 | 0.5643 | 0.8754 | 0.0338 | 0.2037 |
| enhanced_raw_top2mean | 0.4569 | 0.5826 | 0.8831 | 0.0317 | 0.2621 |
| enhanced_remove_pc1 | 0.4300 | 0.5443 | 0.8423 | 0.3286 | 0.0119 |
| enhanced_remove_pc3 | 0.4163 | 0.5295 | 0.8256 | 0.2921 | 0.0037 |
| enhanced_whiten | 0.4321 | 0.5444 | 0.8510 | 0.2434 | 0.0031 |
| enhanced_zscore | 0.4357 | 0.5490 | 0.8555 | 0.3018 | 0.0172 |
| fusion_cellheavy | 0.4376 | 0.5474 | 0.8667 | 0.0612 | 0.1657 |
| fusion_current | 0.4487 | 0.5675 | 0.8801 | 0.0324 | 0.2725 |
| fusion_current_logsumexp | 0.4509 | 0.5784 | 0.8829 | 0.0336 | 0.1888 |
| fusion_current_mean | 0.4482 | 0.5738 | 0.8817 | 0.0338 | 0.1833 |
| fusion_current_median | 0.4413 | 0.5619 | 0.8744 | 0.0352 | 0.1915 |
| fusion_current_top2mean | 0.4551 | 0.5804 | 0.8822 | 0.0331 | 0.2493 |
| fusion_imageheavy | 0.4505 | 0.5735 | 0.8843 | 0.0235 | 0.3585 |
| fusion_nocell | 0.4341 | 0.5514 | 0.8764 | 0.0207 | 0.3992 |
| fusion_noimage | 0.4251 | 0.5268 | 0.8526 | 0.0873 | 0.1404 |
| fusion_textureheavy | 0.4504 | 0.5724 | 0.8834 | 0.0255 | 0.3271 |
| image | 0.4178 | 0.5232 | 0.8657 | 0.0273 | 0.3821 |
| texture | 0.4040 | 0.4935 | 0.8324 | 0.0144 | 0.4805 |

Projection gate candidates: none
