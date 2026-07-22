# CellAtlas v4 第三次组会汇报说明文档

更新时间：2026-07-20

用途：配合《CellAtlas_v4_简明展示稿》制作 PPT 或进行口头汇报。建议汇报时长 12–15 分钟，主体 11 页，另附答疑口径。

## 一、汇报主线

本次汇报属于方法类项目，建议围绕下面的证据链展开：

```text
问题：固定 patch / 固定 superpixel / 静态相似度不足以支持通用视觉提示分割
  ↓
Phase 1：先建立可信的像素分割、数据与多尺度读取基线
  ↓
Phase 2：把固定 SLIC 改为可学习区域
  ↓
Phase 3：把 HE 中的细胞证据注入区域表示
  ↓
Phase 4：检验跨尺度上下文是否真的有效
  ↓
阶段决策：保留 learned region + cell，暂不保留复杂 cross-scale graph
  ↓
下一步：训练正/负视觉 prompt 驱动的 target-vs-rest 分割
```

汇报重点不是“已经做了很多模块”，而是说明每个模块都通过独立消融决定保留或关闭。当前最强证据是细胞分支有效；最重要的负结果是跨尺度图对 region 分类没有明确增益。

## 二、建议 PPT 结构与讲解词

### 第 1 页：研究目标与当前进度

页面标题：**CellAtlas v4：面向 WSI 的通用视觉提示组织分割**

页面内容：

- 输入：多尺度 HE WSI + 正/负视觉 prompt。
- 输出：用户当前目标的 target-vs-rest 组织 mask。
- 目标：模型统一训练一次，推理时用 prompt 定义目标，不针对每个类别重新训练。
- 进度：Phase 1–4 完成；Phase 5 Prompt encoder 与 Phase 6 mask decoder 待完成。

建议讲解：

> v4 的最终目标不是固定预测 12 类，而是让用户通过正负视觉样例定义当前要找的组织。前四个阶段先解决图像表征、区域划分、细胞信息和跨尺度上下文，后两阶段才进入真正的 prompt-conditioned mask 学习。目前前四阶段已经完成正式训练和消融。

### 第 2 页：为什么采用六阶段拆分

页面标题：**先验证表示，再训练 prompt mask**

页面内容：

| 现有问题 | v4 对应模块 |
|---|---|
| 单一局部视野 | 多尺度输入 |
| 固定 SLIC 边界和语义不稳定 | 深度区域化 |
| region token 缺少细胞组成证据 | Cell-to-region |
| 不同视野独立 | 跨尺度交互 |
| 简单 prototype/cosine 难处理多正负例 | Prompt set encoder |
| 独立区域打分缺少上下文 | Context-aware mask decoder |

建议讲解：

> 如果从一开始就端到端训练完整模型，即使结果不好，也无法判断问题来自区域边界、细胞特征、多尺度融合还是 prompt matching。因此我们按模块设置技术 gate，每一阶段都用独立消融决定是否进入主路径。

### 第 3 页：数据基础与实验协议

页面标题：**先保证数据可信和 split 隔离**

页面内容：

- 审计 963 张 WSI，确认 HE/GT 对齐、12 类 RGB 标签、未知像素处理和文件完整性。
- 动态索引 537,419 个 512×512 patch，不落盘重复图像。
- 固定开发队列：140 train / 30 validation / 30 test，病人级互斥。
- 三尺度同中心索引：106,734 patch；10x/5x/2.5x 对应逐渐扩大的物理视野。
- 正式任务遵循单例 → 多步 smoke → DDP → validation → checkpoint → resume。

建议讲解：

> 最早的工作重点不是模型，而是把数据契约固定下来。训练 patch 只保存 WSI、坐标和标签统计，使用时从原图动态读取。这样避免存储大量重复 PNG，也能保证不同阶段复用同一个病人级 split。Phase 1 冻结后对 test 只评估一次；Phase 2–4 没有用 test 选模型。

### 第 4 页：Phase 1——10x 像素分割基线

页面标题：**DeepLabV3 建立可靠的组织分割基线**

页面内容：

| 指标 | 结果 |
|---|---:|
| 采样 validation macro Dice | 0.8621 |
| 完整 validation macro Dice | 0.8634 |
| 完整 validation macro mIoU | 0.7640 |
| 冻结 test macro Dice | 0.8716 |

补充：共 12 类；macro Dice 的正式实现排除 background，对 11 个非背景组织类宏平均。

建议讲解：

> 这个结果说明仅使用 10x HE，DeepLabV3 已经能够完成较可靠的 12 类组织分割。完整 validation Dice 为 0.8634，冻结后的独立 test 为 0.8716。失败主要集中在血液、淋巴细胞聚集、坏死和高边界比例区域，为后续细胞与区域模块提供了明确目标。

注意：不要把 0.862 描述成“12 类准确率”；它是排除 background 后的像素级 macro Dice。

### 第 5 页：Phase 1 多尺度消融

页面标题：**简单多尺度早期融合没有带来稳定收益**

页面内容：

| 输入 | 最佳采样 validation Dice |
|---|---:|
| 10x only | **0.87349** |
| 10x + 5x | 0.87129 |
| 10x + 5x + 2.5x | 0.86495 |

建议讲解：

> 三个尺度以同一个 10x 中心对齐，外层视野分别扩大。我们先用最简单的通道拼接验证上下文上限，但没有观察到稳定提升：10x-only 反而最好。因此后续不默认认为“尺度越多越好”，而是把 10x 作为主路径，并在 Phase 4 用 region token 再检验一次多尺度交互。

表述边界：三尺度早期融合和旧 10x 基线最初并非完全相同候选索引，不能直接用 0.86495 对 0.8621 宣称提升；最终冻结依据是上下文完整索引上的对照方向和后续 Phase 4 结果。

### 第 6 页：Phase 2——学习型区域化

页面标题：**从固定 SLIC 转向 64 个可学习区域**

页面内容：

模型输出：soft assignment + 64×256 region tokens。

| 方法 | Purity | Boundary F1 | Oracle region Dice |
|---|---:|---:|---:|
| Learned | **0.8404** | 0.0551 | **0.5138** |
| SLIC | 0.8114 | **0.0565** | 0.3837 |
| Grid | 0.8244 | 0.0247 | 0.4575 |

建议讲解：

> Phase 2 的关键不是直接输出最终 mask，而是学习更适合作为中间表示的区域。相对 SLIC，learned region 的 purity 提升约 2.9 个百分点，oracle Dice 提升约 13.0 个百分点。这说明即使假设后续分类完全正确，学习区域能提供更高的理论分割上限。不过 boundary F1 仍略低于 SLIC，所以当前成功主要来自语义纯度和区域覆盖，而不是边界指标全面领先。

指标解释：oracle region Dice 是把每个区域赋予其 GT 多数类别后能达到的理论上限，不是当前模型的最终像素分割 Dice。

### 第 7 页：Phase 3——细胞如何进入区域

页面标题：**Cell-to-region attention 注入微观证据**

页面内容：

```text
HE 中的细胞
→ [位置、面积、7类细胞类型、XCellFormer 64维特征]
→ 74维 cell embedding
→ 按 soft assignment 约束的 region-query attention
→ cell-aware region token
```

工程要点：

- 密集 patch 最多保留 255 个空间分层细胞，同时保留真实总细胞数。
- 细胞类别使用 one-hot，不把类别编号误当连续变量。
- 细胞只覆盖 10x fine 视野，未伪装成三个尺度都有细胞特征。

建议讲解：

> 已有 XCellFormer 能直接从 HE 生成细胞特征，因此无需 mIF。我们用 region token 作为 query，从空间上属于该区域的细胞中聚合信息；不仅使用特征，还使用细胞密度。对于超过 255 个细胞的 patch，不再简单截断前 255 个，而是做空间分层采样并保留真实总数，避免密度偏差。

### 第 8 页：Phase 3 细胞消融结果

页面标题：**细胞分支有效，尤其帮助淋巴细胞聚集与坏死**

页面内容：

| Variant | Accuracy | Macro F1 | CE |
|---|---:|---:|---:|
| Cell full | **0.90930** | **0.89118** | **0.23234** |
| Region-only | 0.90132 | 0.88549 | 0.27096 |
| Cell zeroed | 0.89727 | 0.86871 | 0.27575 |

建议在右侧放三条结论：

- 公平收益：cell full vs region-only，accuracy +0.798 pp，macro F1 +0.569 pp。
- 依赖证据：推理时置零细胞，macro F1 下降 2.247 pp。
- 分类别：坏死 +1.84 pp；淋巴细胞聚集 +1.09 pp；血液 -1.82 pp。

建议讲解：

> Region-only 是独立训练、同预算的公平基线；cell-zeroed 则用于回答完整模型是否真的使用细胞。完整模型同时优于公平基线，而且屏蔽细胞后明显下降，两条证据共同支持细胞分支有效。淋巴细胞聚集在屏蔽细胞后 F1 从 0.8731 降到 0.7438，符合该组织依赖细胞组成的预期。血液 recall 出现回退，说明细胞信息不是对所有类别都自动有利。

注意：这里是 region classification，不是像素 Dice。

### 第 9 页：Phase 3 表示空间审计

页面标题：**细胞信息不仅改善分类头，也改善 embedding 几何**

页面内容：

| Variant | kNN Acc | kNN Macro F1 | Class-balanced mAP |
|---|---:|---:|---:|
| Cell full | **0.89599** | **0.87717** | **0.50361** |
| Region-only | 0.87726 | 0.85420 | 0.12352 |

建议讲解：

> 为避免“只是分类头更强”的解释，我们固定同一批 validation region 做 kNN 和 retrieval。加入细胞后 kNN accuracy 提升 1.87 个百分点，macro F1 提升 2.30 个百分点，class-balanced mAP 从 0.1235 提升到 0.5036。这说明细胞分支确实重塑了 region embedding，使同类区域在特征空间中更集中、更可检索。

### 第 10 页：Phase 4——跨尺度交互消融

页面标题：**跨尺度图在 region 分类上未证明收益**

页面内容：

- 正式 cache：90,744 patch，约 9.94 GB，0 failure。
- 每 patch：三组 64×256 tokens + 稀疏父子边与几何信息。

| Variant | Val accuracy | Val CE | vs fine-only |
|---|---:|---:|---:|
| Fine-only | 0.91059 | 0.23702 | — |
| Naive concat | **0.91111** | **0.23034** | +0.052 pp |
| Hierarchical bidir | 0.91047 | 0.23522 | -0.012 pp |

建议讲解：

> Phase 4 再次从 region token 层面检验多尺度。简单拼接只提升 0.052 个百分点，层级双向图略低于 fine-only，均处于约 0.1 个百分点的波动范围。按照预先设定的 improve gate，跨尺度主路径关闭，Phase 5 使用更简单的 fine-only cell-aware tokens。这个负结果避免后续 prompt 模型背负无证据支持的复杂度。

补充口径：本轮主消融只有 region accuracy 和 CE，没有完整 12 类 F1 与 WSI bootstrap，因此结论是“没有明确收益”，而不是证明跨尺度在所有任务上绝对无效。

### 第 11 页：阶段性结论与冻结主路径

页面标题：**保留有证据的模块，关闭无明确收益的模块**

页面内容：

```text
10x HE
→ DeepLabV3/病理图像特征
→ Learned soft regions（64×256）
→ XCellFormer cell-to-region attention
→ Fine cell-aware region tokens
→ Phase 5 Prompt encoder
```

建议列出四项判断：

1. 10x 像素分割基线可靠：validation Dice 约 0.86。
2. Learned region 改善 purity 和 oracle Dice，但边界仍可提升。
3. Cell-to-region 通过消融和 retrieval 双重验证，应保留。
4. Cross-scale graph 未过 improve gate，暂不进入默认主路径。

建议讲解：

> 当前不是把所有模块都叠加，而是得到了一条经过逐步消融筛选的主路径。区域学习解决固定 SLIC 的语义上限问题，细胞分支提供确定的微观增益；跨尺度图没有得到足够证据，因此暂时关闭。这样 Phase 5 的 prompt 实验更容易解释，也更节省训练成本。

### 第 12 页：Phase 5–6 计划

页面标题：**下一阶段：正/负视觉 prompt 驱动的通用二分类分割**

页面内容：

Phase 5：

- 从 GT 自动生成 target-vs-rest episode。
- 输入 1–N 个正 prompt 和 0–N 个负 prompt。
- 正、负集合分别编码，再融合为 task token。
- 输出 prompt-conditioned region scores / task-aware tokens。

Phase 6：

- task token + fine region tokens + soft assignment。
- 输出区域概率并投影到像素 mask。
- 评价 Dice、mIoU、prompt efficiency、困难类别和失败案例。

实验原则：

- 先比较 mean prototype 与 prompt set encoder。
- 比较只用正例与正负例联合输入。
- 比较 1、2、4 个正例以及不同负例数量。
- validation 冻结方案前不查看 test。

建议讲解：

> 下一阶段的关键问题不再是固定 12 类分类，而是模型能否根据当前给定的正负视觉 prompt 定义目标。第一版只使用已经验证有效的 fine cell-aware tokens，先回答 prompt encoder 是否优于简单 prototype。Phase 6 再把区域分数可靠地投影成像素 mask。

## 三、汇报中必须统一的指标口径

| 阶段 | 指标 | 含义 | 不应如何表述 |
|---|---|---|---|
| Phase 1 | Macro Dice / mIoU | 像素级 12 类组织分割；macro Dice 排除 background | 不叫“12 类准确率” |
| Phase 2 | Oracle region Dice | 假设每个区域分类正确时，区域边界能达到的上限 | 不等于最终预测 Dice |
| Phase 2 | Region purity | 单一区域内部 GT 类别一致程度 | 不等于分类准确率 |
| Phase 3–4 | Region accuracy / macro F1 | 有效 region 的 12 类分类 | 不等于像素分割 Dice |
| Phase 3 | Retrieval mAP / kNN | embedding 的可分性与检索能力 | 不等于最终 mask 质量 |

## 四、建议重点展示的三组图

如果后续制作 PPT，优先使用以下已有材料：

1. Phase 1 的 `HE | GT | Prediction | Overlay` 最佳、最差和随机案例，用于说明像素分割基线与困难边界。
2. Phase 2 的 best/worst region overlay，用于展示 learned boundary 是否落在真实组织交界处。
3. Phase 3 三种模型的重点类别对比表或柱状图，突出 lymphocyte aggregate、necrosis 和 blood。

Phase 4 建议用简洁柱状图展示三种 variant，纵轴范围不要从 0 开始隐藏差异，但必须标出绝对数值，并注明差异仅约 ±0.05 pp，避免视觉夸大。

## 五、导师可能追问的问题与回答口径

### 问题 1：Phase 1 已经有 0.86 Dice，为什么还要做 region 和 prompt？

回答：Phase 1 是固定 12 类监督分割，只能输出训练时定义的类别；最终任务要求用户通过视觉 prompt 临时定义 target-vs-rest 目标。Region token 是为了把 WSI 转成可交互、可稀疏计算的中间表示，并支持 prompt 匹配和上下文推理。

### 问题 2：Phase 2 oracle Dice 只有 0.51，是不是效果很差？

回答：它与 Phase 1 的像素 Dice 不是同一口径。Phase 2 使用固定数量的 64 个区域来压缩一个 patch，oracle Dice 衡量这种区域离散化的上限。相对基线更关键：learned region 从 SLIC 的 0.3837 提升到 0.5138，同时 purity 从 0.8114 提升到 0.8404。不过绝对上限仍不高，说明 Phase 6 需要利用 soft assignment 或进一步改善边界，而不能只做 hard region 回填。

### 问题 3：细胞分支只提升 0.8 个百分点，值得保留吗？

回答：收益在公平 region-only 基线上同时体现在 accuracy、macro F1 和 CE；屏蔽细胞会进一步明显退化。更重要的是 embedding kNN macro F1 提升 2.30 个百分点、retrieval mAP 从 0.1235 提升到 0.5036，且淋巴细胞聚集、坏死的改善符合组织学预期。因此它不仅是总体 accuracy 的微小波动，而是表示空间和重点类别都得到改善。

### 问题 4：为什么跨尺度无效？

回答：目前只能确认它在 region 分类任务上没有明确收益。可能原因包括 10x 已覆盖主要判别信息、粗尺度 token 对中心 fine region 的额外信息有限、父子构图或训练目标没有迫使模型利用上下文。当前证据不足以支持复杂模型，因此先关闭；如果 Phase 5/6 的 prompt mask 在歧义区域明显需要上下文，可针对该任务重新打开，而不是沿用当前 region 分类图模块。

### 问题 5：为什么 Phase 4 选 fine-only，而不是数值最高的 naive concat？

回答：naive concat 只比 fine-only 高 0.052 个百分点，低于预设的有效提升门槛，且本轮还没有 WSI paired bootstrap 支持稳定性。复杂度增加但收益处于噪声量级，所以选择更简单、证据更清晰的 fine-only。

### 问题 6：test 是否泄漏？

回答：200 WSI 按病人划分为 140/30/30。Phase 1 在方案冻结后进行过一次独立 test 评估；Phase 2–4 的架构选择、训练与正式消融只使用 train/validation。Phase 5 冻结前继续不查看 test。

### 问题 7：最终方法目前完成了吗？

回答：尚未完成。Phase 1–4 已经完成表示学习和跨尺度消融，但真正的 prompt-conditioned episodic training 与 context-aware mask decoder 位于 Phase 5–6。当前能声明的是上游表示模块的阶段性结论，不能声称完整交互式分割系统已经实现。

## 六、推荐结尾

建议最后用下面这段话收束：

> 本阶段完成了从像素基线到区域、细胞和跨尺度表示的逐层验证。学习区域显著提高了区域纯度和 oracle 上限；细胞注入在分类、重点类别和 embedding 检索上均得到正向证据；跨尺度图在当前 region 分类上没有通过收益门槛，因此未进入默认主路径。下一阶段将以 10x cell-aware fine tokens 为冻结输入，正式验证正负视觉 prompt 能否定义并分割新的 target-vs-rest 任务。

## 七、结果来源

- Phase 1 状态报告：`benchmarks/v4/doc/phase1/PHASE1_STATUS_20260715_130849.md`
- Phase 2 validation：`/nfs-medical3/zyh/v4/phase2/analysis/phase2_validation_analysis_20260717_204100/summary.json`
- Phase 3 阶段报告：`/nfs-medical3/zyh/v4/phase3/reports/phase3_closeout_20260719_174946/PHASE3_CELL_REGION_REPORT_20260719_174946.md`
- Phase 3 embedding closeout：`/nfs-medical3/zyh/v4/phase3/reports/phase3_embedding_closeout_20260719_200126.md`
- Phase 4 cache audit：`/nfs-medical3/zyh/v4/phase4/reports/formal_cache_audit_20260719_223137.md`
- Phase 4 主消融：`/nfs-medical3/zyh/v4/phase4/reports/phase4_main_ablation_20260720_003939.md`
