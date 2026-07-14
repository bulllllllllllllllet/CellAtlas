# Phase C Negative Prompt Controls

- paired hard-negative queries: 1500
- strategies: extreme_hard, positive_only, random_negative, bounded_hard_negative
- bounded hard quantiles: 0.85-0.95

| strategy | scale | n | mAP | AUROC | Dice | mIoU | score std |
|---|---|---:|---:|---:|---:|---:|---:|
| bounded_hard_negative | large | 1500 | 0.5528 | 0.8652 | 0.4659 | 0.3353 | 0.0260 |
| bounded_hard_negative | medium | 1500 | 0.5879 | 0.8886 | 0.5058 | 0.3653 | 0.0315 |
| bounded_hard_negative | small | 1500 | 0.5669 | 0.8821 | 0.4984 | 0.3589 | 0.0381 |
| extreme_hard | large | 1500 | 0.5209 | 0.8344 | 0.4333 | 0.3080 | 0.0229 |
| extreme_hard | medium | 1500 | 0.5667 | 0.8672 | 0.4819 | 0.3444 | 0.0270 |
| extreme_hard | small | 1500 | 0.5302 | 0.8579 | 0.4645 | 0.3299 | 0.0327 |
| positive_only | large | 1500 | 0.5711 | 0.8734 | 0.4731 | 0.3367 | 0.0383 |
| positive_only | medium | 1500 | 0.5859 | 0.8864 | 0.4932 | 0.3512 | 0.0483 |
| positive_only | small | 1500 | 0.5738 | 0.8821 | 0.4935 | 0.3509 | 0.0544 |
| random_negative | large | 1500 | 0.5158 | 0.8598 | 0.4454 | 0.3179 | 0.0363 |
| random_negative | medium | 1500 | 0.5239 | 0.8690 | 0.4618 | 0.3287 | 0.0461 |
| random_negative | small | 1500 | 0.5225 | 0.8702 | 0.4694 | 0.3352 | 0.0525 |
