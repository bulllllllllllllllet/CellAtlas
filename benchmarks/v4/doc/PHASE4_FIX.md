# Phase 4 重接入与修复纲要

## 目标与边界

目标不是证明“多尺度一定提升 Dice”，而是建立一条可训练、可审计的完整
Phase 4 → Phase 5 → Phase 6 路径，并在固定 validation 集上判断跨尺度语义
是否带来稳定收益。

- 使用固定的 200-WSI 开发队列及既定 train/validation 切分。
- 模型选择仅使用固定 4,000-episode validation manifest；不得读取已封存的 test split。
- 所有新运行产物写入新的时间戳目录 `/nfs-medical3/zyh/v4/`，不覆盖既有 J5/J10 结果。
- 当前性能主路仍是 J5 fine-scale；在新的配对验证满足门槛前，不得宣称 Phase 4 提升最终 Dice。

## 当前状态

### 已修复：上行/双向语义重复

`CrossScaleModel` 中的 `hierarchical_up` 已改为严格执行：

```text
Fine → Middle → Coarse
```

`hierarchical_bidir` 执行：

```text
Fine → Middle → Coarse → Middle → Fine
```

已新增回归测试，验证两种变体的准确消息方向，并验证相同权重、相同输入下
两者输出不再完全相同。Phase 4 geometry/model 测试目前共 9 项通过。

### 尚未解决：现有 J10 不是完整 Phase 4

当前 Phase 5 主路使用 Fine-only 的 cell-aware token。J10 仅将旧 cache 中的
Middle/Coarse token 通过静态 overlap gather、残差 MLP 和零初始化 gate 注入在线
Fine token；它没有训练完整的 `HierarchicalBlock`。

因此，J10 的近零变化只能评价“旧父尺度缓存 + adapter”，不能评价修正后的完整
跨尺度表征。

## 当前 token 与细胞特征契约

现有多尺度 cache 的 token 来源为：

```text
Fine (10x)       = Phase-2 visual region token + Phase-3 cell-to-region fusion
Middle (5x)      = Phase-2 visual region token
Coarse (2.5x)    = Phase-2 visual region token
```

只有 Fine 当前使用 cell 特征。Middle/Coarse 不应简单复用 Fine crop 内的细胞；
它们的物理视野更大，应该从同一 WSI 级细胞库中按各自 level-0 crop box 查询更多
细胞。Fine 的细胞集合通常是 Middle 的子集，Middle 又是 Coarse 的子集。

本次第一阶段**不**给 Middle/Coarse 加 cell 分支：先隔离和验证跨尺度图本身的效果。
如果该路径有效，再单独建立 WSI 级细胞空间索引并比较三尺度 cell-aware 的增量。

## 第一阶段：完整静态双向图接入

### A. 在线三尺度编码

修改 Phase 6 联合数据集，使每个 episode 读取空间对齐的 10x、5x、2.5x HE crop。
三种尺度均由当前同一个 Phase-2 checkpoint / 在线参数编码，不再将更新后的 Fine
backbone 与历史 cache 的 Middle/Coarse token 混用。

Fine 继续执行现有 Cell-to-Region attention；Middle/Coarse 暂时保持 visual-only。
从三尺度 soft assignment 在当前 forward 中构建并校验 Fine→Middle、Middle→Coarse
top-4 物理重叠边。

### B. 真实 Phase 4 backbone

在 `JointPromptMaskModel` 中接入 `HierarchicalBlock`，替代 `ParentContextAdapter`：

```text
10x visual + cell fusion ─┐
5x visual token           ├→ corrected cross-scale block → updated Fine token
2.5x visual token         ┘                                  ↓
                                             Phase 5 prompt + Phase 6 decoder
```

`hierarchical_bidir` 的 updated Fine token 是分割主路线；prompt 的坐标、Fine region
标签和 episode manifest 仍以 Fine assignment 为准。

严格上行的 `hierarchical_up` 不回写 Fine，因此不会改变 Fine-mask 输出；它保留为
高尺度表征或 region-classification 对照，不作为 Fine segmentation 的主性能路线。

### C. 配置与 checkpoint 契约

- 新增独立的 Phase 6 配置；不改写 J5/J10 配置或 checkpoint。
- 显式声明 `cross_scale_variant`、block 数、top-k、训练组和每组学习率。
- 从同一个 J5 validation-selected checkpoint 初始化 Fine/Prompt/Decoder 对照；新建的
  cross-scale block 单独记录初始化与可训练参数数目。
- 旧 J10 parent-context state 不加载到完整图模型；仅允许经明确的兼容规则缺失。
- 训练 metadata 必须记录三个尺度的 encoder checkpoint、是否在线编码、边构建版本和
  validation manifest hash。

## 第二阶段：内容相关稀疏 attention

静态版本确认可端到端运行后，将 `gather_parents()` 的固定 overlap 均值替换为稀疏
attention：

```text
child query × top-k parent keys + log(overlap) bias
→ masked softmax over valid parents
→ parent values 的加权和
```

几何 top-k 仍是稀疏 mask / bias，不再是唯一的父节点选择权重。必须保留第一阶段的
静态 overlap 实现，作为同参数预算的对照。

## 第三阶段：三尺度 cell-aware（仅在前两阶段有明确价值时）

建立 WSI 级细胞特征空间索引：

```text
wsi_id, cell_x_level0, cell_y_level0, cell_feature
```

每个尺度按自己的 level-0 crop box 查询细胞，再转换到该尺度 assignment 坐标，分别
调用 Cell-to-Region attention。该阶段必须比较：

```text
Fine cell-aware only
vs.
Fine + Middle + Coarse cell-aware
```

不能将 Fine patch 的局部细胞复制给大视野尺度。

## 验证门禁

### 代码与数据不变量

- 三尺度 crop 中心和 level-0 边界嵌套正确；GT/HE/Fine 输出尺寸一致。
- 每条 cross-scale edge 的索引有效、mask 正确、有效边权归一；shuffled control 仍满足
  相同的形状和权重约束。
- 静态与 attention 路径只访问 top-k 合法父节点。
- Fine cell 坐标与 Fine assignment 对齐；第二阶段之前禁止向 Middle/Coarse 伪造 cell 输入。
- 初始 Fine-only 等价模式（若启用）必须以 max-abs logit/token 审计证明，而非仅凭配置。

### 训练门禁

依次完成单样本 forward、代表性多步 BF16 smoke、validation/checkpoint、resume、DDP
smoke。每个训练组记录有限性、梯度范数、GPU/rank 批次分配和 checkpoint 恢复状态。

跨尺度利用率必须按 epoch 保存：

- 跨尺度残差范数 / Fine token 范数；
- 静态路径的消息范数，或 attention 路径的熵与有效父节点数；
- cross-scale block 梯度范数；
- 若保留 gate，则记录 gate 值。

接近零的残差或 gate 表示未利用，不可直接解释为“架构无效”。

## 正式比较与决策

在同一固定 4,000-episode validation manifest 上比较：

1. Fine-only（J5 参考）；
2. 完整 Phase 4 静态 bidirectional graph；
3. 完整 Phase 4 内容相关 sparse attention；
4. edge-shuffled parent control；
5. 仅在 region-level / 高尺度任务上：corrected upward-only 对照。

所有路线报告逐 episode TP/FP/FN，并做 paired bootstrap（macro/micro Dice 的 95% CI）。
同时报告 Boundary F1、unprompted Dice、prompt-conflict rate、目标类别和 prompt size
分层指标，以及跨尺度利用率。只有相对 Fine-only 的配对结果满足预先冻结的门槛，才可
将该路线提名为新的候选；否则 J5 继续作为主路。

## 实施顺序

1. 完成在线三尺度 Dataset / crop / edge 不变量；
2. 接入静态 `hierarchical_bidir` 到 Phase 5/6 主路径，并补齐 checkpoint/config 契约；
3. 完成单样本、多步、resume、DDP 门禁；
4. 运行 static bidirectional 与 Fine-only 的固定 validation 对照；
5. 实现并验证 sparse attention，再做 static / attention / shuffled 的配对比较；
6. 仅在出现稳定增益后，投入 WSI 级细胞索引与三尺度 cell-aware 消融。

## 相关实现位置

- `benchmarks/v4/phase_4_cross_scale/src/model.py`
- `benchmarks/v4/phase_4_cross_scale/tools/build_multiscale_token_cache.py`
- `benchmarks/v4/phase_4_cross_scale/src/export_tokens.py`
- `benchmarks/v4/phase_5_prompt_encoder/`
- `benchmarks/v4/phase_6_mask_decoder/src/joint_dataset.py`
- `benchmarks/v4/phase_6_mask_decoder/src/joint_model.py`
- `benchmarks/v4/phase_6_mask_decoder/train_joint_pixel.py`
