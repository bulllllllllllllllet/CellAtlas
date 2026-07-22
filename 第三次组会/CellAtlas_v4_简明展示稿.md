# CellAtlas v4 阶段性进展：从多尺度组织分割到细胞感知区域表示

更新时间：2026-07-20

## 1. 项目目标

CellAtlas v4 的目标是构建一个面向全切片病理图像（WSI）的通用视觉提示组织分割系统：模型以多尺度 HE 图像和用户提供的正、负视觉 prompt 为输入，在不针对每个新目标类别重新训练的情况下，输出 target-vs-rest 组织分割结果。

完整流程包含六个模块：

```text
多尺度 WSI 输入
→ 深度区域化
→ 细胞信息注入区域
→ 跨尺度区域交互
→ Prompt encoder
→ Context-aware mask decoder
```

目前 Phase 1–4 已完成正式训练与消融，Phase 5–6 尚未开始。

## 2. 数据与实验基础

- 完成 963 张 WSI 的 HE/GT 配对、12 类标签颜色、空间对齐和异常像素审计。
- 建立动态 patch index，共 537,419 个 512×512 patch；训练时按坐标从原图动态读取，不保存重复 patch 图像。
- 日常开发冻结为 200-WSI 病人级隔离队列：140 train、30 validation、30 test。
- 三尺度采用同中心 10x、5x、2.5x 视野，共 106,734 个上下文完整 patch：74,909 train、15,835 validation、15,990 test。
- test 在 Phase 1 方案冻结后只评估一次；Phase 2–4 的选择与消融均未使用 test。

## 3. Phase 1：多尺度输入与像素分割基线

首先训练 DeepLabV3-ResNet50 进行 12 类组织像素分割，建立后续模块的图像表征和数据读取基线。

- 10x 正式基线最佳 validation macro Dice：**0.8621**，macro mIoU：**0.7625**。
- 完整 validation：16,426 个 patch，macro Dice：**0.8634**。
- 冻结后一次性 test：16,601 个 patch，macro Dice：**0.8716**；该结果未用于调参。
- 困难类别主要是血液、淋巴细胞聚集和坏死，高边界比例 patch 更难。

多尺度输入消融的最佳采样 validation Dice：

| 输入 | 最佳 Dice |
|---|---:|
| 10x only | **0.87349** |
| 10x + 5x | 0.87129 |
| 10x + 5x + 2.5x | 0.86495 |

结论：简单通道拼接没有证明多尺度上下文有效，因此后续冻结 10x 作为主路径；多尺度是否有价值，需要在更匹配上下文推理的任务中重新验证。

## 4. Phase 2：深度区域化

Phase 2 不再把固定 SLIC 当作最终区域，而是学习 64 个 soft region assignment，并输出 256 维 region token。SLIC 只提供初始化/伪监督，同时加入紧致性、纯度和边界相关约束。

在完整 validation 15,835 个 patch 上：

| 方法 | Region purity | Boundary F1 | Oracle region Dice | 活跃区域数 |
|---|---:|---:|---:|---:|
| Learned region | **0.8404** | 0.0551 | **0.5138** | 63.64 |
| SLIC | 0.8114 | **0.0565** | 0.3837 | 27.56 |
| Grid | 0.8244 | 0.0247 | 0.4575 | 64.00 |

结论：学习区域相对 SLIC 的 purity 提升约 **2.90 个百分点**，oracle region Dice 提升约 **13.02 个百分点**，说明区域语义一致性和理论分割上限明显改善；但 boundary F1 未超过 SLIC，边界贴合仍是局限。Phase 2 通过，选用 epoch 28。

## 5. Phase 3：细胞感知区域表示

在冻结 Phase 2 的基础上，从 HE 图像提取细胞，并将细胞位置、面积、7 类细胞类型和 64 维 XCellFormer 特征组成 74 维细胞特征。使用 region token 查询细胞，通过带 soft assignment 空间约束的 attention pooling，将细胞证据注入对应区域。

固定 4,000 个 validation patch、254,504 个有效 region 的联合消融：

| 模型 | Region accuracy | Macro F1 | CE |
|---|---:|---:|---:|
| Cell full | **0.90930** | **0.89118** | **0.23234** |
| Region-only | 0.90132 | 0.88549 | 0.27096 |
| Cell zeroed | 0.89727 | 0.86871 | 0.27575 |

核心证据：

- Cell full 相对公平的 region-only 基线：accuracy **+0.798 个百分点**，macro F1 **+0.569 个百分点**。
- 在完整模型中屏蔽细胞后：accuracy **-1.202 个百分点**，macro F1 **-2.247 个百分点**，证明模型确实使用了细胞证据。
- 坏死 F1：0.8169 vs 0.7986，提升 1.84 个百分点。
- 淋巴细胞聚集 F1：0.8731 vs 0.8623；屏蔽细胞后降至 0.7438。
- 血液 F1 从 region-only 的 0.8262 降至 0.8080，主要原因是 recall 下降，需要继续监控。

Embedding 审计进一步支持该结论：cell full 相对 region-only 的 kNN accuracy 提升 1.87 个百分点，kNN macro F1 提升 2.30 个百分点，class-balanced retrieval mAP 从 0.1235 提升到 0.5036。

结论：细胞分支有稳定且可解释的收益，Phase 3 通过，cell-aware fine token 成为后续主干表示。

## 6. Phase 4：跨尺度区域交互

构建了 train+validation 的正式三尺度 token cache：

- 90,744 个 patch，0 duplicate、0 missing、0 failure。
- 每个尺度均为 64×256 FP16 region token，并保存稀疏父子边、centroid、area 和 active mask。
- 缓存约 9.94 GB，4 GPU 构建耗时约 54.4 分钟。

三种主消融均训练 30 epoch：

| Variant | Best validation region accuracy | Val CE | 相对 fine-only |
|---|---:|---:|---:|
| Fine-only | 0.91059 | 0.23702 | — |
| Naive concat | **0.91111** | **0.23034** | +0.052 个百分点 |
| Hierarchical bidirectional graph | 0.91047 | 0.23522 | -0.012 个百分点 |

结论：简单拼接的增益只有 0.052 个百分点，层级双向图没有超过 fine-only，差异处于约 0.1 个百分点的噪声范围。本轮跨尺度 improve gate 未通过，因此 Phase 5 默认使用 **fine-only：10x Phase 3 cell-aware tokens**，不携带复杂跨尺度图。

这是一个有价值的负结果：跨尺度可能对 prompt-conditioned mask 有帮助，但在当前 region 分类任务中没有足够证据支持增加模型复杂度。

## 7. 当前总体结论

1. 数据、坐标和病人级 split 已形成稳定、可复现的数据基础。
2. 10x DeepLabV3 已建立可靠的 12 类像素分割基线，validation Dice 约 0.86。
3. 学习型区域化显著提高 region purity 和 oracle region Dice，但边界 F1 仍需提升。
4. 细胞信息对区域表示有明确收益，尤其帮助淋巴细胞聚集和坏死。
5. 当前多尺度融合对 region 分类没有明确收益，复杂层级图被暂时关闭。
6. 当前冻结主路径为：**10x → learned regions → cell-aware fine tokens**。

## 8. 下一步

Phase 5 将建立 Prompt encoder：

- 从 GT 自动生成 target-vs-rest episodic task；
- 将正/负点或区域映射到 fine regions；
- 分别用 set encoder 聚合多个正例和负例，再形成 task token；
- 预测 prompt-conditioned region score / task-aware token；
- 先完成单样本、smoke、DDP、validation、checkpoint 和 resume gate，再正式训练；
- 方案冻结前不查看 test。

Phase 6 再将 task token 与 fine region tokens 投影为像素 mask，并评估 Dice、mIoU、prompt efficiency 和失败模式。

## 9. 汇报时的一句话总结

我们已经把 v4 从可靠的 10x 组织分割基线推进到可学习区域和细胞感知表示：学习区域提高了区域纯度与理论上限，细胞分支带来约 0.8 个百分点的 region accuracy 增益；跨尺度图在当前任务上没有证明价值，因此下一阶段将以更简洁的 10x cell-aware 表示进入 prompt-conditioned segmentation。

