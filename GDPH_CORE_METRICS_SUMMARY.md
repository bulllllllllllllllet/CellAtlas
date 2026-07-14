# GDPH Benchmark 核心指标汇总

数据来源：`/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/FINAL_REPORT.md`

## 一句话结论

GDPH 是外部验证集，没有参与 XCellFormer 训练。当前结果显示：原始 CTransPath 细胞特征在细胞级组织分类上最好；XCellFormer 的 `reg` 头在 cell+patch 检索中 mAP/AUROC 最好；单纯细胞表征受细胞覆盖率限制，对 fat、mucus、necrosis 这类低细胞或无细胞区域天然不足，因此后续应做多尺度训练，而不是只优化细胞 head。

## 1. 原分辨率细胞检测有效性

| 指标 | fullres | resized_10x | 核心含义 |
|---|---:|---:|---|
| F1 @ 8 px | 84.31% | 4.08% | 必须在原始 level-0 HE 上检测细胞；10x 缩放后几乎丢失细胞级定位能力 |
| F1 @ 12 px | 85.00% | 4.13% | fullres 检测稳定，说明后续细胞特征评估的输入基础成立 |
| F1 @ 20 px | 87.27% | 4.21% | 宽松阈值下 fullres 仍显著优于 resized_10x |

## 2. 细胞级组织分类

| Head | Pooled accuracy | Pooled mIoU | Macro-F1 | Image mIoU | 结论 |
|---|---:|---:|---:|---:|---|
| raw | 62.39% | 39.43% | 51.20% | 37.49% | 最好；CTransPath 原始细胞特征对 GDPH 组织类别更强 |
| reg | 57.80% | 35.46% | 46.83% | 34.42% | 低于 raw；mIF 对齐没有提升组织分类 |
| proj | 55.49% | 32.67% | 43.98% | 31.62% | 最低；当前 contrastive head 没有带来额外组织语义收益 |

## 3. Query-by-region 细胞检索

| Head | Queries | mAP | AUROC | Binary accuracy | Binary F1 | Binary IoU | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| raw | 1237 | 41.15% | 78.90% | 67.99% | 33.26% | 23.67% | mAP/F1 最好，细胞检索整体更稳 |
| reg | 1237 | 40.30% | 80.34% | 81.76% | 28.82% | 19.22% | AUROC/accuracy 最好，但二值区域质量较弱 |
| proj | 1237 | 37.43% | 77.43% | 72.34% | 29.18% | 19.91% | 低于 raw/reg，当前 projection head 收益有限 |

## 4. 细胞覆盖率与传播上限

| 10x 半径 | Cell coverage | Covered oracle accuracy | Whole-image oracle accuracy | Whole-image oracle mIoU | 含义 |
|---:|---:|---:|---:|---:|---|
| 10 px | 31.75% | 94.42% | 29.99% | 41.52% | 细胞附近预测很准，但覆盖范围太小 |
| 25 px | 49.55% | 87.97% | 43.62% | 52.55% | 覆盖提升，但仍有大面积组织无法由细胞解释 |
| 50 px | 59.87% | 84.89% | 50.98% | 55.02% | 接近可用上限，但牺牲局部精度 |
| 100 px | 72.47% | 83.05% | 60.59% | 55.97% | 覆盖最高，但已经是较粗的细胞邻域传播 |

## 5. 低细胞组织覆盖率

| Class | Coverage @ 25 px | Coverage @ 50 px | Coverage @ 100 px | Mean nearest-cell p90 | 结论 |
|---|---:|---:|---:|---:|---|
| necrosis | 75.24% | 89.54% | 96.62% | 52.7 px | 相对还能被细胞邻域覆盖 |
| mucus | 38.32% | 68.78% | 92.52% | 87.0 px | 需要很大半径才覆盖，细胞信号不足 |
| fat | 31.14% | 57.45% | 82.66% | 125.0 px | 最依赖 patch/区域特征，单细胞路线天然弱 |

## 6. Patch 与 Cell+Patch 检索

| Method | Queries | mAP | AUROC | Binary F1 | Binary IoU | Cell coverage | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| cell_raw_nearest_uncapped | 830 | 51.61% | 82.57% | 37.88% | 26.80% | 57.35% | 细胞-only 中 F1/IoU 最好 |
| cell_reg_nearest_uncapped | 830 | 55.54% | 84.82% | 35.35% | 25.15% | 57.35% | 细胞-only 中 mAP/AUROC 最好 |
| patch_raw | 876 | 54.31% | 82.38% | 8.27% | 5.80% | 57.42% | patch 可补低细胞区域，但当前二值阈值质量差 |
| hybrid_raw | 830 | 57.19% | 86.39% | 21.03% | 14.56% | 57.35% | hybrid 比 cell_raw 提升 mAP/AUROC |
| hybrid_reg | 830 | 59.05% | 86.63% | 15.60% | 10.62% | 57.35% | 总体 mAP/AUROC 最好，但 F1/IoU 不如 cell-only |
| hybrid_proj | 830 | 56.12% | 85.08% | 18.45% | 12.62% | 57.35% | 中间水平 |

## 7. 最核心对比

| 问题 | 最好结果 | 指标 | 解释 |
|---|---|---:|---|
| 细胞检测是否可靠 | fullres | F1 85.00% @ 12 px | 原分辨率检测是必要前提 |
| 细胞级组织分类哪个 head 最好 | raw | mIoU 39.43%, Macro-F1 51.20% | 当前 XCellFormer mIF 对齐没有提升 GDPH 组织语义 |
| 细胞区域检索哪个 head 最好 | raw/reg 接近 | raw mAP 41.15%, reg AUROC 80.34% | raw 更稳，reg 排序分离度略好 |
| cell+patch 综合检索最好 | hybrid_reg | mAP 59.05%, AUROC 86.63% | patch 信息能补充细胞-only 的覆盖盲区 |
| 单细胞方法的结构性瓶颈 | coverage @ 25 px | 49.55% | 只有约一半 tissue 像素能被合理细胞邻域覆盖 |
| 最需要 patch 的组织 | fat/mucus | fat p90 125 px, mucus p90 87 px | 低细胞/无细胞区域不能只靠细胞表征 |

## 8. 对后续训练的直接含义

| 观察 | 对训练的含义 |
|---|---|
| `reg/proj` 都弱于 `raw` 的组织分类 | mIF 对齐目标不足以学习 GDPH tissue semantics |
| `proj` 没明显优于 `reg` | 当前 contrastive head 可能没有形成独立有效的检索空间 |
| `hybrid_reg` 的 mAP/AUROC 最好 | 多尺度信息有价值，应把 patch/region objective 纳入训练，而不是后处理拼接 |
| fat/mucus/necrosis 覆盖不足 | 新训练需要 patch-level 或 region-level tissue supervision |
| cell-only F1/IoU 有时优于 hybrid | patch 分支不能只追求排序 mAP，也要优化空间边界/阈值稳定性 |
