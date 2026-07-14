# Phase D Prompt Scale Selector

- queries: 3704
- feature count: 100
- label/evaluation: raw-GT pixel-level Dice and mIoU; input features exclude GT-derived valid_fraction.
- winner: {'method': 'learned_gbdt', 'label_target': 'formal', 'n': 3704, 'scale_accuracy': 0.6260799136069114, 'macro_f1': 0.32857444540167274, 'mAP': 0.5772094746214328, 'AUROC': 0.8841714569623005, 'PixelDice_classwise_toparea': 0.42928088523175945, 'Pixel_mIoU': 0.2933331655463676, 'PixelBestDice': 0.5153710227758116}

| method | label | Pixel Dice | Pixel mIoU | mAP | AUROC | accuracy | macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed_large |  | 0.3700 | 0.2444 | 0.5741 | 0.8750 |  |  |
| fixed_medium |  | 0.4072 | 0.2737 | 0.6032 | 0.8903 |  |  |
| fixed_small |  | 0.4284 | 0.2931 | 0.5699 | 0.8810 |  |  |
| learned_gbdt | bestdice | 0.4292 | 0.2934 | 0.5782 | 0.8844 | 0.6174406047516199 | 0.32745027079677935 |
| learned_gbdt | formal | 0.4293 | 0.2933 | 0.5772 | 0.8842 | 0.6260799136069114 | 0.32857444540167274 |
| learned_mlp | bestdice | 0.4278 | 0.2924 | 0.5737 | 0.8819 | 0.6085313174946004 | 0.30279805134111176 |
| learned_mlp | formal | 0.4280 | 0.2923 | 0.5761 | 0.8832 | 0.6174406047516199 | 0.31302967234434814 |
| learned_rf | bestdice | 0.4219 | 0.2876 | 0.5949 | 0.8884 | 0.5380669546436285 | 0.41578528907325524 |
| learned_rf | formal | 0.4230 | 0.2882 | 0.5941 | 0.8886 | 0.5545356371490281 | 0.425058803482495 |
| manual_area_rule |  | 0.4040 | 0.2721 | 0.5871 | 0.8841 |  |  |
