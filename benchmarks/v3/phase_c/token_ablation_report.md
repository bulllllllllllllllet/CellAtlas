# Phase C Token Ablation

- queries: 3711
- variants: image_only, cell_reg, image_cell_reg_cellw0p5, image_cell_reg_texture_cellstats, image_patch_cell_reg

| token | scale | prompt | n | mAP | AUROC | BestDice | Dice | mIoU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cell_reg | large | clean | 1293 | 0.5262 | 0.8142 | 0.5295 | 0.4505 | 0.3159 |
| cell_reg | large | hard_negative | 1500 | 0.4885 | 0.8189 | 0.4983 | 0.4122 | 0.2865 |
| cell_reg | large | noisy | 911 | 0.4251 | 0.8048 | 0.4357 | 0.3525 | 0.2367 |
| cell_reg | medium | clean | 1293 | 0.5299 | 0.8144 | 0.5294 | 0.4596 | 0.3227 |
| cell_reg | medium | hard_negative | 1500 | 0.5123 | 0.8355 | 0.5201 | 0.4445 | 0.3112 |
| cell_reg | medium | noisy | 918 | 0.4431 | 0.8055 | 0.4464 | 0.3793 | 0.2555 |
| cell_reg | small | clean | 1293 | 0.4977 | 0.7747 | 0.5016 | 0.4428 | 0.3083 |
| cell_reg | small | hard_negative | 1500 | 0.4874 | 0.8055 | 0.5102 | 0.4461 | 0.3124 |
| cell_reg | small | noisy | 918 | 0.3844 | 0.7594 | 0.4110 | 0.3540 | 0.2374 |
| image_cell_reg_cellw0p5 | large | clean | 1293 | 0.5904 | 0.8716 | 0.5846 | 0.4991 | 0.3578 |
| image_cell_reg_cellw0p5 | large | hard_negative | 1500 | 0.5495 | 0.8598 | 0.5440 | 0.4539 | 0.3247 |
| image_cell_reg_cellw0p5 | large | noisy | 911 | 0.4563 | 0.8377 | 0.4647 | 0.3781 | 0.2583 |
| image_cell_reg_cellw0p5 | medium | clean | 1293 | 0.6072 | 0.8834 | 0.5903 | 0.5157 | 0.3699 |
| image_cell_reg_cellw0p5 | medium | hard_negative | 1500 | 0.5770 | 0.8814 | 0.5669 | 0.4946 | 0.3561 |
| image_cell_reg_cellw0p5 | medium | noisy | 918 | 0.4887 | 0.8487 | 0.4824 | 0.4126 | 0.2805 |
| image_cell_reg_cellw0p5 | small | clean | 1293 | 0.6050 | 0.8814 | 0.5874 | 0.5204 | 0.3730 |
| image_cell_reg_cellw0p5 | small | hard_negative | 1500 | 0.5573 | 0.8772 | 0.5606 | 0.4902 | 0.3520 |
| image_cell_reg_cellw0p5 | small | noisy | 918 | 0.4553 | 0.8444 | 0.4742 | 0.4103 | 0.2785 |
| image_cell_reg_texture_cellstats | large | clean | 1293 | 0.6275 | 0.8894 | 0.6140 | 0.5278 | 0.3841 |
| image_cell_reg_texture_cellstats | large | hard_negative | 1500 | 0.5845 | 0.8783 | 0.5716 | 0.4809 | 0.3491 |
| image_cell_reg_texture_cellstats | large | noisy | 911 | 0.4811 | 0.8492 | 0.4830 | 0.3963 | 0.2750 |
| image_cell_reg_texture_cellstats | medium | clean | 1293 | 0.6454 | 0.8994 | 0.6239 | 0.5461 | 0.3980 |
| image_cell_reg_texture_cellstats | medium | hard_negative | 1500 | 0.6159 | 0.8985 | 0.5967 | 0.5230 | 0.3821 |
| image_cell_reg_texture_cellstats | medium | noisy | 918 | 0.5235 | 0.8642 | 0.5096 | 0.4374 | 0.3014 |
| image_cell_reg_texture_cellstats | small | clean | 1293 | 0.6270 | 0.8932 | 0.6097 | 0.5419 | 0.3934 |
| image_cell_reg_texture_cellstats | small | hard_negative | 1500 | 0.5840 | 0.8902 | 0.5821 | 0.5108 | 0.3714 |
| image_cell_reg_texture_cellstats | small | noisy | 918 | 0.4639 | 0.8486 | 0.4818 | 0.4183 | 0.2850 |
| image_only | large | clean | 1293 | 0.5625 | 0.8694 | 0.5614 | 0.4825 | 0.3420 |
| image_only | large | hard_negative | 1500 | 0.4768 | 0.8350 | 0.4976 | 0.4080 | 0.2872 |
| image_only | large | noisy | 911 | 0.4209 | 0.8255 | 0.4349 | 0.3595 | 0.2435 |
| image_only | medium | clean | 1293 | 0.5697 | 0.8724 | 0.5563 | 0.4879 | 0.3440 |
| image_only | medium | hard_negative | 1500 | 0.4921 | 0.8504 | 0.5081 | 0.4333 | 0.3036 |
| image_only | medium | noisy | 918 | 0.4414 | 0.8526 | 0.4489 | 0.3828 | 0.2575 |
| image_only | small | clean | 1293 | 0.6057 | 0.8848 | 0.5805 | 0.5189 | 0.3719 |
| image_only | small | hard_negative | 1500 | 0.5068 | 0.8584 | 0.5249 | 0.4520 | 0.3213 |
| image_only | small | noisy | 918 | 0.4332 | 0.8507 | 0.4495 | 0.3939 | 0.2673 |
| image_patch_cell_reg | large | clean | 1293 | 0.5454 | 0.8460 | 0.5461 | 0.4694 | 0.3297 |
| image_patch_cell_reg | large | hard_negative | 1500 | 0.5149 | 0.8436 | 0.5152 | 0.4335 | 0.3005 |
| image_patch_cell_reg | large | noisy | 911 | 0.4241 | 0.8159 | 0.4397 | 0.3585 | 0.2414 |
| image_patch_cell_reg | medium | clean | 1293 | 0.5643 | 0.8611 | 0.5580 | 0.4880 | 0.3437 |
| image_patch_cell_reg | medium | hard_negative | 1500 | 0.5381 | 0.8617 | 0.5353 | 0.4673 | 0.3267 |
| image_patch_cell_reg | medium | noisy | 918 | 0.4395 | 0.8239 | 0.4508 | 0.3797 | 0.2543 |
| image_patch_cell_reg | small | clean | 1293 | 0.5583 | 0.8610 | 0.5550 | 0.4957 | 0.3491 |
| image_patch_cell_reg | small | hard_negative | 1500 | 0.5213 | 0.8617 | 0.5300 | 0.4706 | 0.3291 |
| image_patch_cell_reg | small | noisy | 918 | 0.4056 | 0.8194 | 0.4368 | 0.3769 | 0.2507 |
