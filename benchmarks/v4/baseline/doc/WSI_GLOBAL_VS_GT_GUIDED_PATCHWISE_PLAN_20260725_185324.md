# WSI 全图分割与 GT 引导逐 patch 基线方案

## 1. 实验目的

本实验比较两种完成 WSI 目标组织分割的方式：

1. **本文方法**：由一次种子提示发起，在整张 WSI 上预测目标组织掩膜；
2. **SAM、SAM-Med2D、WSI-SAM**：先由 GT 确定哪些 patch 含有目标组织，再在每个含目标 patch 的 GT 目标区域内随机采样提示点，独立预测局部掩膜，最后按空间坐标拼接为 WSI 掩膜。

该方案的目的不是模拟真实人工逐 patch 标注流程，而是给局部交互式模型提供准确的 patch 级目标定位，评估其在 **oracle/GT-guided local prompting** 条件下的分割能力。由于 patch 是否含目标及正点位置均使用 GT，该实验必须在论文中明确标注为“GT 引导的局部提示上界”，不能解释为实际人工交互成本或零样本全图泛化能力。

## 2. 冻结的 WSI 任务

在验证集的 30 个 WSI 上，对每个冻结的 WSI—目标类别对建立一项任务。每项任务包含：

- WSI 标识、目标类别、完整 10x patch 网格和 WSI 坐标；
- 本文方法使用的种子 patch、1 个正点和 1 个负点；
- 每个 patch 的 GT 有效像素数、目标像素数和 `has_target` 标记；
- 对所有 `has_target=true` patch 生成的固定随机正/负点；
- 随机种子、GT/patch manifest 哈希、点坐标及生成版本。

任务 manifest 在正式推理前冻结。GT 仅参与构造 baseline 的 oracle prompt manifest 及最终评价；模型推理阶段只读取图像和冻结的点坐标。

## 3. GT 引导逐 patch 提示生成

对第 \(i\) 个 patch，设有效像素为 \(V_i\)，目标二值 GT 为 \(Y_i\)。

1. 若 \(|Y_i \cap V_i|=0\)，则标记该 patch 为 `has_target=false`；不调用 SAM 类模型，拼接时写入全零预测；
2. 若 \(|Y_i \cap V_i|>0\)，则标记为 `has_target=true`；
3. 从 \(Y_i \cap V_i\) 的像素中心中，以冻结伪随机种子均匀抽取 1 个正点；
4. 从 \(V_i \setminus Y_i\) 的像素中心中，以同一规则均匀抽取 1 个负点。若该集合为空，则记录 `negative_available=false`，仅提供正点；不得以 GT 外的临时规则替代；
5. 同一 patch、同一类别和同一随机种子必须得到完全相同的点坐标。建议随机种子定义为 `global_seed + hash(wsi_id, target_class, patch_id)`；
6. SAM、SAM-Med2D 和 WSI-SAM 对每个 `has_target=true` patch 接收相同的正负点、图像尺寸和二值化阈值。

正点随机采样而非取质心，避免人为选择“最容易”的组织中心；随机种子固定可保证实验可复现。建议同时保存每个点是否落在 GT 目标/非目标区域的审计结果。

## 4. 方法执行与 WSI 拼接

### 4.1 本文方法：WSI 级推理

本文方法从种子 patch 的 1 个正点和 1 个负点出发，对完整 WSI patch 网格执行全图推理，并将概率图回填至共同的 10x WSI 坐标系。非重叠网格直接拼接；若采用重叠网格，必须在 manifest 中冻结 overlap、概率融合和阈值规则。本文方法不得使用 patch-level `has_target` 标签或 oracle 点。

### 4.2 SAM 类方法：GT 引导局部推理

对每个 `has_target=true` patch，SAM、SAM-Med2D 或 WSI-SAM 执行一次局部 `predict` 调用，得到概率图和二值掩膜；对 `has_target=false` patch 写入空掩膜。所有 patch 输出按其 WSI 坐标回填，构成完整 WSI 预测。对于 SAM 的多候选输出，按预先冻结的候选选择规则选出一个掩膜，但仍计为一次 `predict` 调用。

## 5. 必须报告的 Oracle 成本与调用次数

对每个 WSI—类别任务，记完整网格 patch 数为 \(N_{patch}\)，含目标 patch 数为 \(N_{target}\)。对每种 SAM 类模型，必须记录：

| 字段 | 定义 |
|---|---|
| `N_patch` | 完整 WSI 覆盖的 patch 数 |
| `N_target` | GT 指示含目标组织的 patch 数 |
| 正点数 | `N_target` |
| 负点数 | 至多 `N_target`，缺少负像素的 patch 单独计数 |
| **SAM 调用次数** | **精确等于 `N_target`** |
| 未调用空 patch 数 | `N_patch - N_target` |
| 总预测覆盖 | 全部 `N_patch` 的拼接掩膜 |

结果表必须报告每种 SAM 类模型的：全验证集总调用次数 \(\sum N_{target}\)、每 WSI 的中位数/均值/95% 分位数调用次数，以及每 WSI 的正负点总数。因为 `has_target` 本身来自 GT，该数量是 **oracle patch-localization cost**，不是实际人工成本；不能将其与本文方法的一次 WSI 查询直接作为真实标注效率比较。

## 6. 指标

在完整拼接后的 WSI 掩膜上、仅对有效像素计算：

- WSI 微平均 Dice；
- WSI 宏平均 Dice；
- 类别宏平均 Dice；
- mIoU；
- Boundary F1@2px；
- Disconnected-region recall：对不含本文方法种子正点的 GT 连通区域，预测覆盖率 \(\geq 50\%\) 时计为召回；
- Exclude-seed-region Dice：排除种子正点所在 GT 连通区域后，在其余目标区域上计算 Dice；
- WSI 覆盖率：验证预测是否覆盖完整有效 WSI 坐标范围。

对于 SAM 类基线，Disconnected-region recall 和 Exclude-seed-region Dice 仍可报告，但解释时应强调其使用了每个含目标 patch 的 oracle 正点；这些指标反映局部掩膜质量与拼接质量，而不是从种子提示跨空间检索的能力。

## 7. 建议结果表

### 表 1：GT 引导逐 patch 上界下的 WSI 分割精度

| 方法 | WSI 微 Dice | WSI 宏 Dice | 类宏 Dice | mIoU | Boundary F1 | DRR | Exclude-seed-region Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| 本文方法（1 个 WSI 种子提示） |  |  |  |  |  |  |  |
| SAM（GT 引导逐 patch） |  |  |  |  |  |  |  |
| SAM-Med2D（GT 引导逐 patch） |  |  |  |  |  |  |  |
| WSI-SAM（GT 引导逐 patch） |  |  |  |  |  |  |  |

### 表 2：Oracle prompt 与模型调用审计

| 方法 | WSI 任务数 | 总 patch 数 | 总含目标 patch 数 | 总正点数 | 总负点数 | **总模型调用次数** | 调用/WSI（中位数） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 本文方法 |  |  | — |  |  |  |  |
| SAM（GT 引导逐 patch） |  |  |  |  |  |  |  |
| SAM-Med2D（GT 引导逐 patch） |  |  |  |  |  |  |  |
| WSI-SAM（GT 引导逐 patch） |  |  |  |  |  |  |  |

表题和正文必须包含“GT-guided”或“oracle-prompted”字样。不能把表 2 的点击和调用次数描述为真实人工标注时间。

## 8. 执行门禁

1. 先在 1 个 WSI—类别任务上生成 oracle prompt manifest，逐点验证正点落在目标、负点落在非目标，且所有 patch 坐标在界内；
2. 使用同一 WSI 分别运行三个 SAM 类模型，验证每个含目标 patch 恰有一次调用、每个无目标 patch 为零掩膜且零调用；
3. 将局部输出拼接回 WSI 后，验证图像、GT、预测尺寸及坐标范围一致；
4. 通过单 WSI gate 后运行全验证集；逐 WSI 持久化预测、TP/FP/FN、点数、调用次数和失败记录；
5. 汇总前检查 `sum(model_calls) == sum(N_target)`，以及完整 WSI 覆盖率为 100%。

## 9. 论文中的准确表述

推荐表述：

> 为评估局部提示分割模型在准确目标定位条件下的上限性能，我们构建了 GT-guided patch-wise prompting protocol。对于每个含目标组织的 patch，从其目标 GT 区域随机采样一个正点，并从非目标有效区域随机采样一个负点；SAM 类模型在这些 patch 上独立推理，其输出随后拼接为 WSI 掩膜。该协议依赖 oracle patch 定位，因此结果反映局部掩膜质量上界，而不代表仅凭一次用户点击即可完成的真实全图交互能力。我们同时报告总提示点数和总模型调用次数。

对于本文方法，可在同一段后说明其只接收种子 patch 的一次正负提示，不读取其他 patch 的 GT 存在性或 oracle 点。
