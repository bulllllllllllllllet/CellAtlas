# v4 Dice 指标与 prompt 评估口径

更新时间：2026-07-24 CST

本文汇总 Phase6 最终冻结 test 的 Dice 口径，以及 whole-slide inference 的当前评估状态。

## 1. 评估单位

一个 **query / episode** 不是单纯的一个物理 patch，而是：

| 构成 | 含义 |
| --- | --- |
| patch | 一个 10x 图像 patch |
| target class | 12 个组织类别之一；任务为该类 vs 其余类别 |
| positive / negative prompts | 正、负提示的 region 集合 |
| prompt size | `point`、`small` 或 `large` |

同一 patch 可以因目标类别或 prompt 集不同而产生多个 query。

## 2. Dice 的计算

在像素阈值 0.5 后、忽略 `ignore_index` 像素，单个 query 的 Dice 为：

`Dice = 2TP / (2TP + FP + FN)`。

| 指标 | 聚合方式 | 当前最终 test 结果（J5） |
| --- | --- | ---: |
| Pixel macro Dice | 每个 query 各算一次 Dice，再对全部 query 等权平均 | 0.739237 |
| Pixel micro Dice | 汇总全部 query 的 TP、FP、FN 后再计算 Dice | 0.812773 |

因此 macro 不是严格先对 12 个类别等权平均；它是对 query 等权平均。评估采样会按“类别 × prompt size × 数据组”近似均衡，使类别权重接近、但不保证完全相等。

## 3. 最终冻结 test 的覆盖范围

| 项目 | 数值 |
| --- | ---: |
| test split 完成预处理的 patch | 15,990 |
| 可行 prompt episodes | 131,818 |
| 最终 test 抽样 query occurrences | 4,000 |
| 实际不同 episode index | 3,931 |
| 实际不同 patch | 3,175 |
| 像素阈值 | 0.5 |

最终 test 是从可行 episodes 进行分层均衡抽样，并非对 15,990 个 test patch 各跑一次。

## 4. 按 prompt size 的 Pixel macro Dice

| Prompt size | Query 数 | J3 baseline | J5 final | J5 − J3 |
| --- | ---: | ---: | ---: | ---: |
| point | 1,552 | 0.641542 | 0.645285 | +0.003743 |
| small | 1,444 | 0.772883 | 0.772904 | +0.000021 |
| large | 1,004 | 0.836852 | 0.836047 | -0.000805 |
| 混合总体 | 4,000 | 0.737979 | 0.739237 | +0.001258 |

`point` 不是整体最终分数的全部来源；最终 test 按约 `40% / 35% / 25%` 的 point/small/large 比例混合评估。

## 5. 12 类结果的位置

最终 test 的 12 类分项位于：

`/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_test_20260722_134900/summary.json`

读取 JSON 的 `by_target_class.<class_id>.joint.pixel` 即为 J5 在该 target class 下的 query-level 平均 Pixel Dice；`baseline.pixel` 是 J3 基线。

逐 query 的 TP/FP/FN 在同目录 `episode_metrics.parquet`，可用于重新计算 macro、micro 或自定义分层统计。

## 6. Whole-slide inference 的 Dice 状态

| 项目 | 当前状态 |
| --- | --- |
| 整图推理 | 已对 1 张 validation WSI 完成 2 次 prompt 条件的全 tile 拼接推理 |
| WSI prompt 机制 | source tile 的正/负 region token 被编码为全局 task token，并应用到整张 WSI 的每个 tile |
| 整图 Dice 数值 | 尚未产出 |
| 原因 | 已完成 tile、coverage、mask 与 prompt 映射的运行检查；尚未对拼接 mask 运行 GT 对齐的整图 Dice 脚本 |

whole-slide 的 GT 对齐二值 Dice 实现位于 `benchmarks/v4/whole_slide_inference/visualize_wsi_gt_prediction.py`；它计算的是单张 WSI、单个目标类别和单组 prompt 下的整图 Dice，不能与上面的 4,000-query patch-level macro Dice 混为一谈。

## 7. 各模块的已验证贡献

下表严格区分了“加入模块”的消融和“完整模型中继续微调该模块”的后期对照。不同实验的任务和评估口径不同，不能横向当作同一条 Pixel Dice 曲线。

| 模块 / 对照 | 对比口径 | 结果 | 结论 |
| --- | --- | --- | --- |
| Phase1 多尺度输入 | 固定数据与坐标入口，无可训练参数 | 未做单独性能消融 | 基础设施，不单独归因 Dice |
| Phase2 Deep Region Encoder（DeepLab-ResNet50） | 未保留“无 Phase2”同口径对照 | 无独立加入增益 | 是全部后续模块的视觉 region token 来源 |
| Phase3 cell-to-region，加入 cell | 254,504 个 validation region；`cell_full` vs 独立 `region_only` | accuracy +0.007980；macro-F1 +0.005690 | cell 模块本身有明确正收益 |
| Phase3 cell-to-region，去除 cell 输入 | 同一 cell 模型，`cell_full` vs `cell_zeroed` | accuracy +0.012023；macro-F1 +0.022469 | 细胞输入的贡献最明显，约 +2.25 个 macro-F1 点 |
| Phase3，后期继续微调 cell | 固定 360 prompt episodes；J3 cell+decoder vs decoder-only | Pixel macro +0.000128；micro -0.000068 | 已有 cell 模块继续调参没有稳定 Pixel Dice 收益 |
| Phase5 prompt encoder，加入模块 | 没有“移除 prompt 后仍完成同一交互任务”的有效对照 | 无直接增益数值 | prompt 是 target-vs-rest 任务定义的必要条件；后期局部解冻的 best micro +0.001343、macro -0.001224（360 episodes） |
| Phase6 mask decoder，加入 decoder | 4,000 validation episodes；Phase5 baseline → Phase6 joint（早期 cached-target 口径） | Pixel macro 0.72121 → 0.74133（+0.02012）；micro +0.00521；Boundary F1 +0.06299 | 最大的已记录像素级增益；但 region / unprompted 指标下降 |
| Phase2 backbone `layer4` 微调（J5） | 4,000 validation episodes；J3 → J5 | macro 0.732503 → 0.734125（+0.001622）；micro 0.804538 → 0.807057（+0.002519） | 后期最稳定的额外 Pixel Dice 增益 |
| Phase4 parent-context adapter（J10） | 全预算固定 4,000 validation occurrences；J10 − J5 paired comparison | macro +0.000184；micro -0.000977；micro 0.001 非劣的 paired CI 未通过 | 提升太小且不稳定，未覆盖 J5 主路 |

目前正式推理主路仍是：**Phase2 visual region token + Phase3 cell fusion + Phase5 prompt conditioning + Phase6 mask decoder**，并使用 J5 的 `layer4` 微调权重。Phase4/J10 仅保留为研究候选。

## 来源

- 最终 test 汇总：`/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_test_20260722_134900/summary.json`
- 最终 test 逐 query 明细：`/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_test_20260722_134900/episode_metrics.parquet`
- 联调总台账：`benchmarks/v4/doc/联调方案/README.md`
