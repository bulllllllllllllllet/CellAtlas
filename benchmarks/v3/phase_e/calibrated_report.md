# Phase E Calibrated MLP Mask Decoder

Strict 12/4/4 WSI-level train/calibration/test protocol.

| method | n | Pixel Dice | Pixel mIoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| fixed_0p5 | 3711 | 0.4255 | 0.2925 | 0.3539 | 0.7332 |
| global_calibrated | 3711 | 0.4337 | 0.2978 | 0.4015 | 0.6341 |
| classwise_calibrated | 3711 | 0.4654 | 0.3230 | 0.4470 | 0.6071 |
| phase_c_fixed_small | 3711 | 0.4277 | 0.2926 | 0.5036 | 0.4372 |
