# Phase C Pixel Metrics

- prediction: selected full superpixels from the formal LOIO classwise area prior.
- GT: original raw GT pixels; only annotated pixels covered by a superpixel are evaluated.
- computation: segment-wise pixel counts, mathematically identical to rasterizing masks.
- overall: {"PixelDice_classwise_toparea": 0.4014450603786888, "Pixel_mIoU": 0.2701053892219655, "PixelBestDice": 0.49252561474548034, "PixelPrecision": 0.4845776640066579, "PixelRecall": 0.41156534511375453}
