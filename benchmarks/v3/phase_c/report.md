# Phase C Multiscale Baseline Retrieval

## Summary

Phase C computes prompt-conditioned cosine retrieval scores for each Phase B query on small, medium, and large superpixels.
GT hard/soft labels use Phase A standardized `superpixels.csv` fields for speed and consistency with Phase B prompt sampling.

The formal run uses `tokens_image_cell_reg_texture_cellstats.npy`. Phase B hard negatives are sampled from the 85%-95% similarity quantile among valid different-class segments, rather than from the extreme top-similarity tail.

## Outputs

- query-scale scores: /nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/evaluations/query_scale_scores
- metrics CSV: /nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/evaluations/multiscale_baseline_metrics.csv
- validation JSON: /nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/reports/phase_c_validation.json

## Validation

- passed: True
- total_metric_rows: 11133
- ok_metric_rows: 11126
- score_npz_files: 11133
- scale_counts: {'large': 3704, 'medium': 3711, 'small': 3711}

## Mean Metrics

| scale | rows | mAP | AUROC | Dice_classwise_toparea | BestDice |
|---|---:|---:|---:|---:|---:|
| small | 3711 | 0.5693 | 0.8809 | 0.4707 | 0.5669 |
| medium | 3711 | 0.6033 | 0.8903 | 0.4746 | 0.5846 |
| large | 3704 | 0.5741 | 0.8750 | 0.4463 | 0.5646 |

## Token And Prompt Decisions

- Token ablation evaluated 3,711 queries x 3 scales x 5 variants. The enhanced token was the best overall: mAP 0.5822, AUROC 0.8821, LOIO Dice 0.4950, compared with base image+cell mAP 0.5518, AUROC 0.8677, LOIO Dice 0.4712.
- On the formal current-area protocol, enhanced-token overall Dice is 0.4639 and mIoU is 0.3304.
- `image_patch_cell_reg` underperformed the base token, so the current nearest-patch raw feature is not used as the default.
- Paired negative-prompt controls showed that extreme hard negatives compress the score range. The bounded 85%-95% similarity protocol improves over extreme hard negatives and is the formal default.

## Pixel-Level Evaluation

The primary segmentation metric is now computed by rasterizing each selected full superpixel and comparing it directly with the raw GT tissue pixels. Evaluation is restricted to annotated pixels covered by a superpixel. This avoids the majority-label bias against mixed large superpixels and the optimistic bias of the GT-presence diagnostic.

| scale | rows | Pixel Dice | Pixel mIoU | PixelBestDice |
|---|---:|---:|---:|---:|
| small | 3711 | 0.4277 | 0.2926 | 0.5135 |
| medium | 3711 | 0.4065 | 0.2733 | 0.4972 |
| large | 3704 | 0.3700 | 0.2444 | 0.4668 |
| overall | 11126 | 0.4014 | 0.2701 | 0.4925 |

- Pixel-level outputs: `evaluations/multiscale_pixel_metrics.csv` and `evaluations/pixel_class_scale_summary.csv`.
- The 0.0911 gap between PixelBestDice (0.4925) and the LOIO area-prior Dice (0.4014) shows threshold/area selection remains useful, but score ordering and superpixel boundary resolution are the larger remaining limitations.
- GT-presence Dice is retained only as a coverage diagnostic and is not comparable to the pixel-level main metric.

## Next Step

Use the enhanced-token Phase C scores as frozen retrieval input for Phase D scale-selection labels. Phase D uses pixel-level Dice/BestDice labels and WSI-level splits; mAP/AUROC remain auxiliary ranking metrics. Then train the Phase E MLP mask decoder.
