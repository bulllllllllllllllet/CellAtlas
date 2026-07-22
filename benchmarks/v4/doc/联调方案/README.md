# CellAtlas v4 六模块联合训练方案与状态

> 本文档是 Phase1–Phase6 联调的唯一状态台账。每次修改联调代码、配置、训练顺序、验收门槛或 checkpoint 选择规则时，必须在同一次变更中更新本文档。

最后更新：2026-07-22 CST

## 状态定义

- 已完成：实现和对应验收均已通过，且产物路径已记录。
- 进行中：已开始实现或验证，但尚未通过本阶段全部门槛。
- 未开始：尚未修改代码或启动实验。
- 阻塞：已确认存在外部环境或数据契约问题，需要先解决阻塞项。

## 当前六模块主路

| 模块 | 当前作用 | 联调状态 | 备注 |
| --- | --- | --- | --- |
| Phase1 多尺度 WSI 输入 | 提供空间对齐 HE/GT 和多尺度索引 | 已完成 | 固定数据入口，无可训练参数 |
| Phase2 Deep Region Encoder | 像素特征、soft assignment、fine region token | 已完成；J5 最终候选已冻结 | J5 仅解冻 ResNet50 layer4；正式 4000 validation 通过最终 Pixel Dice 门槛 |
| Phase3 Cell-to-region | 将 XCell 细胞信息注入 region token | 已完成；当前冻结 | 后续逐步解冻 fusion/attention |
| Phase4 Cross-scale | fine/middle/coarse 父子上下文 | 正式主路候选；全预算接入中 | 旧J4近似持平而非明确退化；按用户确认，从全预算J5最佳点零门控接入并做非劣验证 |
| Phase5 Prompt Encoder | 正负 prompt set 编码与 target-vs-rest matching | 已完成；当前冻结 | 作为后续防遗忘 teacher |
| Phase6 Mask Decoder | 局部图上下文、region logits、soft pixel projection | 已完成；J5 最终 test 通过 | test Pixel macro/micro=0.739237/0.812773，均过最终门槛且相对J3提升 |

## 已完成的联调基线

### Phase2→3→5→6 fine-only 像素联调

状态：已完成

- 正式训练：30/30 epochs，EXIT_CODE=0。
- 最优 checkpoint：/nfs-medical3/zyh/v4/phase6/joint_runs/phase6_joint_pixel_20260721_172701/checkpoint_epoch_024.pth
- 训练路径：/nfs-medical3/zyh/v4/phase6/joint_runs/phase6_joint_pixel_20260721_172701
- 未读取 test。

独立 4000 validation episodes 结果：

| 指标 | Phase5 基线 | Phase6 联调 | 变化 |
| --- | ---: | ---: | ---: |
| Region macro Dice | 0.84559 | 0.80004 | -0.04555 |
| Unprompted region Dice | 0.81782 | 0.78745 | -0.03036 |
| Pixel macro Dice | 0.72121 | 0.74133 | +0.02012 |
| Pixel micro Dice | 0.80981 | 0.81501 | +0.00521 |
| Boundary F1@2px | 0.22870 | 0.29168 | +0.06299 |

结论：Phase6 明显改善像素边界，但 region 和 unprompted region 出现遗忘，后续目标是保留边界收益并恢复区域语义。

### 联调独立可视化

状态：已完成

- 单例六联图已通过：/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260721_205235
- 完整 4000-episode 指标分片：/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260721_205424/shards
- 完整正式审计与 48 张可视化：/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260721_213820
- 已生成 point/small/large/hard-case 四张 contact sheets、36 张代表图、12 张 hard cases、episode_metrics.parquet、mismatch_details.parquet、summary.json 和 metadata.json，EXIT_CODE=0。
- baseline prompt remap：positive changes=0、negative changes=0、conflicts=0；substantial high-purity mismatch=0。
- 旧 joint epoch24 在 assignment 移动后出现 280 个 positive/negative prompt slot conflicts；这是 J2 必须解决的训练信号。
- 稳定性硬门槛中的“高纯度区域”定义为 purity≥0.90 且有效面积≥1024 pixels；所有更小或低纯度差异仍逐条落盘，但不把 1–2 pixel 新激活 slot 的 purity=1 误判为稳定组织区域违约。


动态 online target/remap 下的正式结果：

| 指标 | Phase5 baseline | 旧 joint epoch24 | 变化 |
| --- | ---: | ---: | ---: |
| Region macro Dice | 0.84565 | 0.78677 | -0.05889 |
| Unprompted region Dice | 0.81822 | 0.77849 | -0.03973 |
| Pixel macro Dice | 0.72124 | 0.73156 | +0.01033 |
| Pixel micro Dice | 0.80982 | 0.80173 | -0.00809 |
| Boundary F1@2px | 0.22866 | 0.28775 | +0.05909 |

说明：前一张表是旧 cached-target 口径；本表是 J1 后的动态 online target/remap 口径，后续 J2 以本表作为正式基线。

## 后续联合训练路线图

### J1：在线 region target 与动态 prompt 重映射

状态：已完成

目标：

1. 根据当前 online assignment 和 pixel GT 动态生成 region target，不再用旧 cache label 直接监督可变 assignment。
2. prompt 保留归一化 region-centroid 坐标，每次 forward 与 online assignment 导出的当前 region centroid 做动态匹配并重新聚合 prompt token。现有 eligibility 未保存内部像素点击或 dense region mask，因此禁止把质心误当成内部点击直接做 assignment argmax。
3. 输出 online positive/negative/prompted region mask，用于 region loss 和 unprompted 指标。
4. 显式记录正负 prompt 映射冲突、slot 改变数和 cached-vs-online target 差异，不静默修补。

验收门槛：

- 单元测试覆盖坐标取样、重复 slot、padding mask、online target 和梯度。
- 单例与多步 smoke 中 loss/梯度有限，Phase2 assignment/embedding 均获得梯度。
- 初始 checkpoint 上高纯度 prompt 的正负关系不变；冲突数必须单独报告。

完成证据：

- 16 个 Phase6 单元测试全部通过。
- 修正后单例：/nfs-medical3/zyh/v4/phase6/joint_runs/phase6_joint_pixel_20260721_212205，EXIT_CODE=0。
- 两轮单卡 BF16 smoke：/nfs-medical3/zyh/v4/phase6/joint_runs/phase6_joint_pixel_20260721_212350，EXIT_CODE=0。
- resume smoke：/nfs-medical3/zyh/v4/phase6/joint_runs/phase6_joint_pixel_20260721_212545，从 epoch 2 继续，EXIT_CODE=0。
- 双卡 DDP smoke：/nfs-medical3/zyh/v4/phase6/joint_runs/phase6_joint_pixel_20260721_212702，EXIT_CODE=0。

### J2：防遗忘稳定化联调

状态：进行中

- 起点：Phase6 joint epoch 24。
- 当前主路仅训练 Phase6 decoder，Phase2 backbone/embedding/assignment 全部冻结；移动 geometry 的旧 J2/J2b/J2c 路线已被对照否决。
- 冻结 DeepLab backbone、Phase3 cell encoder 和 Phase5 prompt encoder。
- 加入冻结 Phase5 teacher 的 region-logit/task 约束，重点保护 unprompted regions。
- 加入可微 prompt-separation loss 和 signed prompt consistency；旧 joint epoch24 的正式动态审计发现 280/4000 episodes 的正负 prompt 映射到同一在线 region，正式训练前必须在 smoke 中证明冲突率下降且不产生 assignment collapse。
- 初始损失建议：pixel BCE 1.0 + pixel Dice 1.0 + boundary 0.1 + online region 0.5 + logit distillation 0.25 + unprompted consistency 0.1 + assignment regularization。
- 预计 8–10 epochs，但必须经过单例、smoke、DDP 和 resume 门禁后才能正式训练。

当前进度：

- [x] 新增 phase6_joint_stabilization.yaml，保持原正式配置不变。
- [x] 支持 --initial-joint-checkpoint 仅 strict 加载旧 joint 模型权重，与 --resume 互斥。
- [x] 实现 cached Phase5 teacher logits/task、稳定 unprompted-region distillation、signed prompt consistency 和可微 prompt-separation loss。
- [x] 新增损失及动态重映射相关单元测试通过；后续迭代后 Phase6 当前共 32 个测试。
- [x] J2 单例：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260721_223431，2 train steps + validation + checkpoint，EXIT_CODE=0；新增损失和三组梯度均有限。
- [x] prompt conflict 指标接入训练、overall validation、12 类和 point/small/large；同时记录冲突 slot、冲突 episode、episode 总数和真实 episode rate。Phase6 共 20 个测试，Phase5 回归 8 个测试，全部通过。
- [x] 多类别/多尺寸 BF16 smoke 已运行并统计训练前后 prompt conflict rate：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_002515，单卡 BF16、3 epochs、每轮 256 train + 固定 360 validation episodes，EXIT_CODE=0。
- [x] online region target 单类 episode 契约修复：pixel GT 含目标并不保证 hard online region majority 同时有正负 region；J2 region auxiliary loss 现在只平均实际存在的类别项，ranking 仅统计同时有正负 region 的 episode，并持久化可评估 episode 数。Phase5 原严格训练契约不变。
- [ ] prompt conflict 门禁未通过：固定 validation 三轮均为 23/360=6.3889%，point/small/large 分别恒为 3.1250%/5.3846%/11.7647%；当前 separation loss 只出现训练内波动，尚未推动 validation hard remap 改变。
- [x] 训练前后同集几何诊断：/nfs-medical3/zyh/v4/phase6/evaluation/prompt_conflict_geometry_20260722_004020，比较旧 joint epoch24 与 J2 epoch0/1/2，EXIT_CODE=0。训练前为 21/360=5.8333%，J2 三轮反而均为 23/360=6.3889%；相对初始发生任一 hard slot 改变的 episode 为 4/5/6，但没有消除冲突。
- [x] J2b hard conflict margin targeted 对照：新增独立 phase6_joint_stabilization_conflict_margin.yaml，不修改原 J2 配置；单轮同预算运行 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_005105，EXIT_CODE=0。22 个训练冲突 pair 进入 margin loss，但固定 validation 仍为 23/360，门禁失败，未继续三轮。
- [x] train epoch0 冲突索引审计：/nfs-medical3/zyh/v4/phase6/evaluation/prompt_conflict_geometry_train_20260722_010210，定位 22 个唯一冲突 dataset indices；选择 14289 作为固定 overfit episode。
- [x] 分量梯度归因：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_011020。episode 14289 上未加权 conflict-margin 的 Phase2 assignment 梯度为 1.0201，但原权重0.1后仅约0.102，相对 configured total 9.9568 约1%；原 soft separation 加权后更低，仅约0.016。
- [x] conflict-margin 权重 1/5/10 单 episode overfit：weight1 运行 20260722_012020 将 margin 从0.0576降至0.0063但未跨 hard boundary；weight5 运行 20260722_012650 在 epoch4 解除冲突；weight10 运行 20260722_013320 在 epoch3 解除并于 epoch4 的全部16 steps保持零冲突。三档均未出现 assignment 正则突变。
- [x] J2c weight10 多 episode smoke + resume：epoch0 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_014020 将固定 validation 从训练前21降至20；从其 epoch0 恢复到新目录 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_014850，确认 start_epoch=1，但 epoch1/2 均反弹到23，稳定性门禁失败。
- [x] J2c 冲突集合迁移：/nfs-medical3/zyh/v4/phase6/evaluation/prompt_conflict_geometry_val_20260722_015650。epoch0 仅解决旧冲突45378且没有新增；epoch1/2保持该项已解决，但新增123518和123523（后者在固定 sampler 重复两次），证明是跨 episode 边界漂移，而非原困难冲突计数不变。
- [x] cached Phase2 safe-geometry anchor 被否决：梯度审计 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_020230 后使用 margin10 + anchor1000 单轮运行 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_020850，validation 为22/360，劣于训练前21和无 anchor 的20。cached Phase2 几何不是旧 joint epoch24 的实际起点，禁止继续该方向。
- [x] 冻结旧 joint geometry teacher 已实现：Phase2 暴露共享 backbone feature 入口，teacher 仅复制并严格加载旧 joint epoch24 的 embedding/assignment，训练 forward 不重复 backbone；teacher 不进入主模型 state_dict，必须通过 --geometry-teacher-checkpoint 显式提供，正 anchor 权重禁止无 teacher fallback。Phase2 6 项、Phase6 21 项测试全部通过。
- [x] teacher 语义与梯度门禁：冲突 episode 14289 审计 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_010929 中 online 与 teacher 的 xy/area 最大差均为0，teacher 正确将其排除在 safe anchor 外，未加权 conflict-margin assignment 梯度仍为1.0201；漂移后的 teacher-safe episode 审计 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_011049 证明 teacher anchor 可向 embedding/assignment 回传非零梯度。两者 EXIT_CODE=0。
- [x] 旧 joint centroid/area teacher anchor 被否决：按梯度定标使用 margin10 + anchor100000 的单轮 256/360 BF16 smoke /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_011210，EXIT_CODE=0，validation 仍为23/360。固定集合诊断 /nfs-medical3/zyh/v4/phase6/evaluation/prompt_conflict_geometry_val_20260722_011448 确认只解决旧冲突45378，同时仍新增 teacher-safe 的123518和123523，与无 teacher anchor 的迁移模式相同。
- [x] teacher-safe Voronoi-gap anchor 被否决：约束每个安全正/负 prompt 对 teacher slot 的最近距离优势，teacher 起点损失/梯度为0，只在 gap 缩小时激活；正常 batch 梯度审计 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_012252 得到未加权 assignment 梯度0.02883，据此选择 weight30。单轮 256/360 BF16 smoke /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_012417 仍为23/360，EXIT_CODE=0；停止继续扫权重或增加 epoch。
- [x] decoder-only 控制实现：新增 phase6_joint_stabilization_decoder_only.yaml，冻结 Phase2 embedding/assignment、Phase3 cell 和 Phase5 prompt，只训练1254401个 decoder 参数；仅作用于 geometry 的 assignment/separation/conflict-margin/anchor 权重归零。训练器现在按配置动态校验实际启用的参数组和非零梯度。Phase2 6 项、Phase6 22 项测试全部通过。
- [x] decoder-only 单例与梯度门禁：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_013229，2 train episodes + 1 validation episode，configured-total decoder 梯度0.65389，BF16/validation/checkpoint均正常，EXIT_CODE=0。
- [x] decoder-only 同集基线：旧 joint epoch24 在固定360 validation episodes 上的只读审计为 /nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260722_013604，Pixel/Region/Unprompted/Boundary 分别为0.74465/0.79989/0.78932/0.28834，冲突21/360，EXIT_CODE=0。
- [x] decoder-only 三轮控制与 resume：epoch0 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_013357；从其 checkpoint 恢复的 epoch1–2 位于 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_013846，run_metadata 确认 start_epoch=1，全部 EXIT_CODE=0。三轮冲突均严格保持21/360；相对旧 joint 基线，epoch0 的 Pixel/Region/Unprompted/Boundary 变化为+0.00022/+0.00263/+0.00064/-0.00024，epoch1为+0.00016/+0.00398/-0.00027/-0.00222，epoch2为+0.00005/+0.00317/+0.00076/-0.00077。epoch0 是当前最均衡 checkpoint，继续 epoch 没有一致收益。
- [x] 冲突 episode 策略已显式化：decoder-only 默认配置保持不过滤；独立 `phase6_joint_stabilization_decoder_only_filter_conflicts.yaml` 仅在训练 loss 中排除冲突 episode，同时分开记录原始输入、排除数、实际优化数和全冲突 batch 数；validation 始终保留原始样本和冲突率。
- [x] 过滤控制的代表性门禁：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_015232，32 train + 64 validation，validation 原始冲突5/64，stress set 恰好5行，EXIT_CODE=0。
- [x] 同预算过滤对照：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_015349，256 train + 固定360 validation，EXIT_CODE=0。训练输入冲突23/256，23个全部排除，233个安全 episode 参与优化，2个全冲突 batch 显式跳过；validation 仍为原始21/360。该结果是消融对照，不自动替代默认主路。
- [x] validation 冲突 stress set 已按 epoch 单独持久化：上述同预算对照产物为 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_015349/stress_set_epoch_000.parquet，21行/21个唯一 episode，与原始 validation 冲突21/360严格一致；保留 episode、WSI、class、size、正负 prompt 几何和冲突 slot。
- [x] 推理 abstention API 已实现：`src.conflict_policy.build_inference_response` 在正负 prompt 映射冲突时返回 `status="abstain"`、明确 reason/message 与冲突 slots，不返回 pixel mask，提示用户调整或分离 prompt 后重试。
- [x] Dice 主导 checkpoint 策略已实现：最终 Pixel macro≥0.72、Pixel micro≥0.7987 为质量硬门槛；Region≥0.84、Unprompted≥0.81、Boundary≥0.2867和原始冲突率≤7%为 soft targets，继续进入 Pareto 和 warning，但不一票否决。合格前沿以 decoder-only 固定基线为参照，按 `max(macro gain, micro gain)` 选优；任一 Pixel Dice 提升即可晋升候选，另一项只需保持硬门槛。冲突率同时解读为推理 abstention coverage。macro 0.72 是用户在 test 解封前确认的最终门槛；此前 0.7383 的实验判定保留为历史记录。
- [x] Pareto 真实路径代表性门禁：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_020620，2 train + 1 validation，decoder-only 梯度有限，checkpoint/last/stress/Pareto 产物齐全，EXIT_CODE=0。两次在导入前失败的 tmux 启动 20260722_020438/020535 已保留 `EXIT_CODE=1` 日志，未产生训练目录；最终统一改为 Python package 入口。
- [x] decoder-only 双卡 DDP smoke：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_020928，显式绑定物理 GPU 0,5，32 train + 固定64 validation，world_size=2，EXIT_CODE=0。两 rank 均通过非零 decoder 梯度联合检查；stress shards 为1+4行，合并5行，与原始 validation 冲突5/64一致；所有 checkpoint/Pareto 产物齐全。该 run 在旧五项硬门槛下因 Boundary=0.28471和5/64冲突产生 `no_eligible_checkpoint`；新 Dice 主导规则下其 Pixel macro=0.76992、micro=0.82607 均合格，Boundary/冲突仅为 soft warnings。64-sample 冲突率不用于硬阻断。
- [x] Dice 主导规则真实路径：最小门禁 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_021935，EXIT_CODE=0；固定256 train + 360 validation 运行 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_022144，EXIT_CODE=0。后者 Pixel macro=0.74487、micro=0.80316均过线，`best_checkpoint.json` 正确指向 epoch0；Region=0.80252、Unprompted=0.78996仅产生 soft warnings，Boundary=0.28810和冲突21/360持续报告；stress set 21行严格一致。
- [ ] 通过全部门禁后再决定是否启动 8–10 epoch 正式 stabilization。

当前三轮 smoke 摘要：

| epoch | train separation loss | train conflict rate | val conflict rate | Pixel macro Dice | Region macro Dice | Unprompted Dice | Boundary F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.03822 | 8.5938% | 6.3889% | 0.74644 | 0.80632 | 0.79160 | 0.28729 |
| 1 | 0.02792 | 6.6406% | 6.3889% | 0.74682 | 0.80949 | 0.79346 | 0.28656 |
| 2 | 0.03153 | 9.3750% | 6.3889% | 0.74627 | 0.80775 | 0.79433 | 0.28660 |

过滤对照与同集基线（固定360 validation episodes）：

| 方案 | Pixel macro Dice | Pixel micro Dice | Region macro Dice | Unprompted Dice | Boundary F1 | 原始 val conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧 joint epoch24 | 0.74465 | 0.80143 | 0.79989 | 0.78932 | 0.28834 | 21/360 |
| decoder-only，不过滤 | 0.74487 | 0.80316 | 0.80252 | 0.78996 | 0.28810 | 21/360 |
| decoder-only，训练过滤冲突 | 0.74502 | 0.80325 | 0.80446 | 0.79126 | 0.28747 | 21/360 |
| J3 cell+decoder，不过滤 | 0.74499 | 0.80309 | 0.80420 | 0.78982 | 0.28810 | 21/360 |
| J3 partial-Phase5+decoder，epoch0 | 0.74465 | 0.80370 | 0.80486 | 0.78960 | 0.28795 | 21/360 |
| J3 partial-Phase5+decoder，最佳 epoch1 | 0.74364 | 0.80450 | 0.80632 | 0.78947 | 0.28631 | 21/360 |
| J4 parent-context adapter，最佳 epoch0 | 0.74367 | 0.80446 | 0.80997 | 0.79312 | 0.28560 | 21/360 |
| J5 backbone layer4，最佳 epoch1 | 0.74594 | 0.80916 | 0.81936 | 0.79520 | 0.28443 | 23/360 |

结论：移动 Phase2 geometry 的 margin/teacher-anchor 路线都会产生跨 episode 冲突迁移，当前全部否决。decoder-only 把冲突稳定在旧 joint 的21/360，仍是冻结 geometry 的 J2 对照主路。按用户确认的最终任务优先级，任一 Pixel Dice 相对当前候选基线提升即可晋升，另一项只需保持硬门槛；Region、Unprompted、Boundary、冲突/abstention 均保留为诊断软目标。J5 在正式 4000-episode 对照中相对 J3 的 Pixel macro/micro 分别提升 +0.001622/+0.002519，macro=0.734125、micro=0.807057 均通过最终 0.72/0.7987 门槛，因此冻结为最终候选；像素阈值固定0.5。冻结时尚未读取test；冻结后唯一一次正式test结果见“最终test”章节。

### J3：逐步解冻 Phase3 和 Phase5

状态：进行中；Phase3 单轮对照已停止，局部 Phase5 三轮对照已完成并晋升候选

- [x] 新增 `phase6_joint_j3_cell_finetune.yaml`，Phase2 geometry 全冻结、Phase5 冻结，只解冻 Phase3 `CellToRegionAttention` 与 Phase6 decoder，cell LR=1e-5。
- [x] 单元梯度门禁通过：Phase2/Phase5 无梯度，cell/decoder 有梯度；Phase6 共30项测试通过。
- [x] 最小真实 smoke：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_022754，configured-total decoder/cell 梯度为0.6539/1.9972，EXIT_CODE=0。
- [x] 固定256/360单轮对照：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_022900，EXIT_CODE=0。相对 decoder-only，Pixel macro +0.000128、Pixel micro -0.000068、Region +0.001686、Unprompted -0.000134、Boundary +0.000007，冲突不变为21/360；stress set严格一致。
- [x] 因 Pixel micro 与 Unprompted 轻微退化，不续 epoch、不做 DDP，J3 Phase3 第一步暂不晋升为主路。
- [x] 局部 Phase5 解冻实现：`phase6_joint_j3_prompt_finetune.yaml` 仅训练 matcher、task_projection、Set Encoder 最后一层与 decoder，prompt LR=1e-5；Phase2 geometry/Phase3 冻结。冻结 Phase5 teacher 是独立快照，不进入学生 state_dict，与学生 optimizer 完全隔离。
- [x] 局部 Phase5 参数范围/teacher/梯度单测通过，Phase6 共32项测试；最小真实 smoke /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_025302 中 configured-total decoder/prompt 梯度为0.6539/4.0519，EXIT_CODE=0。
- [x] 局部 Phase5 固定256/360单轮对照：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_025358，EXIT_CODE=0。相对decoder-only，Pixel macro -0.000212、Pixel micro +0.000546、Region +0.002340、Unprompted -0.000353、Boundary -0.000142，冲突不变21/360；stress set严格一致。
- [x] 旧 joint epoch24 中 Phase5 的65个 tensor 与原 Phase5 epoch27 逐项完全相同，max abs=0；因此已完成 run 中 teacher source 元数据虽写 Phase5 checkpoint，实际快照数值与 initial-joint 一致。后续代码已改为显式记录 initial-joint 快照来源和时机。
- [x] 按“任一 Pixel Dice 提升即可晋升”的新规则完成三轮可恢复对照：epoch0 位于 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_030235，epoch1–2 从该 checkpoint 恢复到 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_030634，run_metadata 确认 start_epoch=1，EXIT_CODE=0。最佳 epoch1 的 Pixel macro/micro 为0.743643/0.804499，相对 decoder-only 为-0.001224/+0.001343，两项均过硬门槛，`best_checkpoint.json` 正确指向 epoch1。冲突仍为21/360，Region/Unprompted/Boundary 未达软目标只记 warning。
- [x] 首次恢复启动 20260722_030459 因 tmux 未继承仓库 Python 包路径在 import 阶段退出，保留 `EXIT_CODE=1` 日志，未进入训练；改用 package 入口后正常完成。
- [x] 局部 Phase5 双卡 DDP 门禁：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_031143，显式绑定物理 GPU 0,5，32 train + 64 validation，world_size=2，EXIT_CODE=0。两 rank 均通过 decoder/prompt 非零梯度联合检查；stress shards 合并5行，与原始 validation 冲突5/64一致，checkpoint/Pareto/best 产物齐全。64-sample Dice 仅用于路径健康门禁，不参与固定360集合排名。
- 建议学习率：decoder 1e-4，assignment 2e-5，embedding 5e-6，cell fusion 1e-5，prompt matcher 1e-5，set encoder 5e-6。
- DeepLab backbone 仍冻结，预计 8–10 epochs。

### J4：Phase4 多尺度 parent context 对照

状态：已完成；epoch0 以微小 Pixel macro 改善纳入候选

- [x] A 为 J3 partial-Phase5 最佳 epoch1；B 在其上只训练 198145 个 parent-context adapter 参数，decoder、Phase2 geometry、Phase3 和 Phase5 全部冻结。
- [x] adapter 使用 fine→middle、middle→coarse top-4 父边汇聚，通过零初始标量 gate 残差接入 fine token。单测和真实初始审计均确认 token max abs=0、gate=0；Phase6 共34项测试通过。
- [x] 真实 cache 契约：90744 条 train/val，三尺度 token 均为[64,256]，两级边为[64,4]，权重归一、索引合法、数值有限。
- [x] 最小梯度审计 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_031721，configured-total parent-context 梯度范数0.07324，EXIT_CODE=0。32/64 多步 BF16 smoke /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_031916 的16个step梯度持续有限非零，validation/checkpoint/stress/Pareto 完整，EXIT_CODE=0。
- [x] checkpoint/resume 门禁：epoch0 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_032018，epoch1 恢复到 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_032119，确认 start_epoch=1，gate、optimizer、scheduler和history连续，均 EXIT_CODE=0。
- [x] 固定256/360三轮对照 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_032220，EXIT_CODE=0。最佳epoch0的Pixel macro/micro为0.743674/0.804459，相对J3最佳为+0.000032/-0.000040，两项均过硬门槛，冲突保持21/360；epoch1–2双Dice均下降，因此停止扩展到5 epochs。
- [x] 双卡DDP门禁 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_032603，物理GPU 0,5，world_size=2，两rank均通过parent-context非零梯度检查，stress shards合并5行与原始validation冲突5/64一致，EXIT_CODE=0。

### J5：有限端到端微调

状态：已完成；按最终 0.72/0.7987 门槛通过，冻结为最终候选

- [x] 从 J3 partial-Phase5 最佳 epoch1 起步，仅解冻 `phase2.backbone.layer4` 的14964736个参数，LR=1e-6；BatchNorm 保持 eval，embedding/assignment、cell、prompt、decoder 权重全部冻结。
- [x] 保留 assignment balance/entropy/compactness 和 prompt teacher logit/task 防遗忘约束，max grad norm=5。单测确认只有 layer4 得到梯度，Phase6 共35项测试通过。
- [x] 最小梯度审计 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_033658，configured-total layer4 梯度13.9401，Pixel/Region/Boundary/三项assignment正则均可回传，EXIT_CODE=0。teacher-logit未加权梯度51.989，经0.25权重和全局裁剪受控。
- [x] 32/64 多步 BF16 smoke /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_033803，16个step的layer4梯度约8.96–15.14，全部有限并裁剪到5；validation冲突5/64，checkpoint/stress/Pareto完整，EXIT_CODE=0。
- [x] checkpoint/resume 门禁：epoch0 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_033928，epoch1 恢复到 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_034025，确认start_epoch=1、layer4 optimizer/scheduler/history连续，均EXIT_CODE=0。
- [x] 双卡 DDP 门禁 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_034156，物理GPU 0,5，world_size=2，两rank均通过layer4非零有限梯度检查，stress shards合并5行与原始validation冲突5/64一致，EXIT_CODE=0。
- [x] 三轮代表性预实验 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_114423 的epoch2达到macro/micro=0.746038/0.809167，但该命令的cosine T_max=3，末轮LR已归零，因此不做伪恢复并不作为最终五轮结论。
- [x] 正确五轮 cosine 对照 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_114809，固定256 train + 360 validation，EXIT_CODE=0。最佳epoch1的Pixel macro/micro=0.745938/0.809163，相对J3最佳同时提升+0.002296/+0.004664；Region=0.81936、Unprompted=0.79520、Boundary=0.28443仅产生soft warnings，冲突23/360=6.3889%仍低于7%软目标。epoch2–4 Dice回退，停止扩展到8 epochs，`best_checkpoint.json`正确指向epoch1。
- [x] 正式双卡 4000-episode validation-only 对照 /nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260722_122219，GPU 0/5、world_size=2、EXIT_CODE=0、test_used=false。J3→J5 的 Pixel macro 0.732503→0.734125（+0.001622），micro 0.804538→0.807057（+0.002519）；J5 在12类中提升8类，point +0.005107，small/large分别-0.000486/-0.000733。该结果按当时0.7383门槛未晋升；用户在test解封前将最终macro门槛调整为0.72后，J5通过0.72/0.7987双门槛并正式晋升。Region/Unprompted macro分别+0.011379/+0.009936，Boundary -0.001105和冲突284/4000=7.10%只记soft warning。`stress_set.parquet`恰284行，与冲突episode严格一致。
- [x] 评估器明确 cache-reference 语义：只有未传入 baseline joint checkpoint 的原始 Phase2/5 cache reference 才对高纯度 cache mismatch 硬失败；显式 J3/J5 checkpoint 间对照仍完整记录 mismatch/remap，但不把已训练 geometry 相对旧 cache 的预期差异误判为无效。旧语义失败运行 20260722_121359 及其全部分片原样保留；修正后 Phase6 共36项测试通过。

### J6：全局像素阈值校准

状态：已完成；收益不足，停止扫描

- [x] 按 `doc/first_talk.md` 的“固定0.5或简单校准”约束，实现单一全局 threshold grid；禁止 classwise threshold、P90 或类别面积先验。每个阈值同时持久化 overall、12类和 point/small/large Pixel Dice，选择规则为 micro≥0.7987 时最大化 macro。Phase6 专项测试通过。
- [x] 同路径单例 /nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260722_123339，7个阈值、98个指标字段、strict load、stress/mismatch、test隔离均通过，EXIT_CODE=0。
- [x] 64-episode 双卡 DDP smoke /nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260722_123513，world_size=2，stress 5/5，EXIT_CODE=0。小样本 J5 最优阈值0.45仅用于证明排序路径健康。
- [x] 正式 4000-episode validation-only 校准 /nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260722_123639，EXIT_CODE=0、test_used=false。J5 最优全局阈值为0.475，Pixel macro/micro=0.734266/0.806286；相对0.5仅改善macro +0.000141、micro -0.000771，macro仍距0.7383硬底线0.004034。J3在0.475为0.732518/0.803640，J5仍相对提升+0.001748/+0.002647。不继续细扫阈值，不读取test。

### J7：layer4 + decoder 联合适配

状态：已完成小规模对照；收益不足，不进入正式4000

- [x] 新增 `phase6_joint_j7_layer4_decoder.yaml`，从 J5 epoch1 起步，只训练 ResNet50 layer4 与 Phase6 decoder，LR分别2.5e-7/2.5e-5；embedding/assignment、cell、prompt、parent-context和BN运行统计冻结，损失配方保持J5不变。
- [x] 参数范围与真实梯度门禁通过：/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_124252，decoder/layer4参数量为1254401/14964736，configured-total梯度范数0.5024/19.6623，BF16、strict initialization、checkpoint、stress/Pareto均正常，EXIT_CODE=0。
- [x] 32 train + 64 validation 多步 BF16 smoke /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_124416，16个step两组梯度持续有限非零，stress 5/5，EXIT_CODE=0。
- [x] 双卡DDP门禁 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_124747，world_size=2，两rank梯度联合校验通过，stress shards 1+4=5，EXIT_CODE=0。
- [x] 固定256 train + 360 validation 三轮可恢复对照：epoch0 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_124929，epoch1–2从其checkpoint严格恢复到 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_125109，start_epoch=1，optimizer/scheduler/scaler/history连续，均EXIT_CODE=0。最佳epoch0 macro/micro=0.746433/0.808258，相对J5为+0.000494/-0.000905，两项仍过硬底线；epoch1/2未扩大macro收益，冲突恒为23/360且stress一致。
- [x] 由于+0.000494远小于J6正式4000 macro距硬底线的0.004034，停止J7的正式4000、续epoch和扩大训练；不读取test。

### J8：逐 episode Dice loss 定标

状态：已完成；无 Dice 收益，停止

- [x] 新增 `phase6_joint_j8_macro_dice.yaml`，与J7相比只将逐episode `pixel_dice` 权重从1.0提到2.0；模型、sampler、layer4/decoder LR和其余损失完全不变，embedding/assignment等geometry heads继续冻结。配置差异契约审计通过。
- [x] 真实单例与梯度定标 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_125652，Dice unit decoder/layer4梯度为0.1562/2.3533，configured-total为0.6190/21.4051，数值有限且受norm=5裁剪保护，EXIT_CODE=0。
- [x] 32 train + 64 validation 的16-step BF16 smoke /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_125755，两组梯度持续有限非零，stress 5/5，EXIT_CODE=0。
- [x] 三轮cosine horizon的固定256 train + 360 validation epoch0 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_125923，macro/micro=0.746430/0.808235，相对J7最佳为-0.000003/-0.000023；冲突23/360与stress严格一致，EXIT_CODE=0。按预设门禁停止续epoch、DDP和正式4000，不读取test。

### J9：训练 episode 覆盖度扩大

状态：已完成；无足够 Dice 收益，不进入正式4000

- [x] 新增 `phase6_joint_j9_coverage.yaml`，与J7相比只将每轮train预算从256提到2048，validation固定360；模型、loss、sampler、LR和三轮cosine计划不变。配置差异契约审计通过，J7已有单例、DDP和resume门禁支撑该尺度提升。
- [x] 双卡三轮运行 /nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_130254，GPU0/5、world_size=2、每轮2048 train + 360 validation，三个immutable checkpoint、Pareto、metrics和stress shards齐全，EXIT_CODE=0、test_used=false。
- [x] epoch0/1/2 Pixel macro/micro分别为0.745977/0.808903、0.745848/0.808924、0.745484/0.808651；最佳epoch0相对J5仅+0.000039/-0.000259，远低于进入正式4000的预设macro门槛0.749972。冲突episode为23/22/22，每轮stress合并行数与之严格一致。
- [x] 停止J9正式4000和更大预算。J6–J9已分别排除全局阈值、layer4+decoder联合适配、Dice权重和训练覆盖度作为填补当前正式macro缺口的简单方案；最终checkpoint规则未冻结前不读取test。

## 最终 test（一次性冻结评估）

状态：已完成；J5 通过最终 Pixel Dice 门槛

- [x] test 前最终候选已冻结：J5 epoch1 `/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_114809/checkpoint_epoch_001.pth`，Pixel threshold=0.5，macro/micro硬门槛=0.72/0.7987。不可变清单为 `/nfs-medical3/zyh/v4/phase6/final_selection/final_candidate_20260722_131830/final_candidate.json`，记录 checkpoint/config/validation summary/stress set 的 SHA-256，且 `test_evaluated=false`。
- [x] 独立test前处理完成：token cache `/nfs-medical3/zyh/v4/phase4/data/multiscale_token_cache_20260722_132500_test` 为15,990/15,990、5卡、failure_lines=0；fine labels `/nfs-medical3/zyh/v4/phase4/data/fine_region_labels_20260722_133300_test` 为15,990行和15,990唯一patch；eligibility `/nfs-medical3/zyh/v4/phase5/data/prompt_eligibility_20260722_133700_test` 扫描15,990 patches并得到131,818个可行episodes。三者均明确`split=test/test_used=true`、EXIT_CODE=0。
- [x] 显式`--split/--cell-routing`评估入口先在validation单例 `/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260722_134601_val_entry_smoke` 通过，`split=val/test_used=false`，J3/J5均strict load；此前脚本路径启动在import前失败且未创建评估目录，改用仓库规定的Python package入口后通过，未读取test。
- [x] 唯一一次正式test：`/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_test_20260722_134900`，GPU0/5、world_size=2、4000 episodes、threshold=0.5、EXIT_CODE=0、`split=test/test_used=true`。J3→J5 Pixel macro 0.737979→0.739237（+0.001258），micro 0.809926→0.812773（+0.002847）；J5 同时通过0.72/0.7987门槛。
- [x] test 分项完整报告：J5在12类中提升7类；point +0.003743、small +0.000021、large -0.000805。J5 Region macro=0.814348、Unprompted macro=0.802735、Boundary F1=0.293499，均作为诊断项保留。
- [x] test冲突未过滤：J5冲突233/4000=5.825%，`stress_set.parquet`恰233行且233个唯一episode；推理继续显式abstain，不返回mask并提示调整prompt。封存结果 `/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_test_20260722_134900/final_test_result.json` 链接pre-test freeze、test summary/stress/log哈希，状态`passed`、`no_post_test_selection=true`。

## 全预算J5 → J10 Phase4接入

状态：进行中；全预算J5已完成，J10配置已绑定其最佳validation checkpoint

- [x] 用户确认Phase4可以在Pixel Dice实质不倒退时进入主路。旧J4在固定256/360上相对J3为macro +0.000032、micro -0.000040，冲突不变；该量级视为近似中性证据，不再解释成结构无效。
- [x] 两阶段正式方案冻结。阶段A为全预算J5控制：从J3 partial-Phase5最佳epoch1起步，只训练ResNet50 layer4，LR=1e-6，5 epochs × 20,000 train episodes，固定4,000 validation episodes。配置为 `configs/phase6_joint_j5_full_budget.yaml`，输出根为 `/nfs-medical3/zyh/v4/phase6/formal_full_runs`。
- [x] 阶段B为J10：从阶段A最佳checkpoint起步，新建零初始化Phase4 parent-context adapter；接入时J5输出严格等价，只训练约198,145个adapter参数，其他模块（包括layer4、geometry、cell、prompt、decoder和BN统计）全部冻结，parent LR=2.5e-5。J10最终配置中的Dice reference必须在阶段A结束后写入其固定4000 validation结果，禁止提前使用旧参考值或test值。
- [x] J10 checkpoint非劣策略已接入通用Pareto代码：相对全预算J5 reference，Pixel macro/micro各允许最多0.001的实用等价差，且至少一项必须正增益；同时继续满足绝对macro≥0.72、micro≥0.7987。Region、Unprompted、Boundary、冲突率和point/small/large/12类继续报告。
- [x] 最终晋升使用同一固定validation episode集合的逐episode配对比较；trainer内的0.001非劣门槛是第一层筛选，阶段B结束后还必须生成配对差值/置信区间报告。若无checkpoint满足非劣规则，则保留全预算J5，不以结构完整性强行覆盖Dice结果。
- [x] 已完成test继续保持不可变封存。阶段A/B的训练、checkpoint选择、LR或停止判断只允许读取train/validation；不再次运行当前test，也不根据其分项结果调整Phase4。
- [x] 全预算J5门禁完成：Phase6 41项测试通过；单例 `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_141959_j5_full_single` 只有14,964,736个layer4参数可训练，configured-total梯度13.9401且各主要损失路径有限；32/64多步BF16 `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142100_j5_full_bf16` 的16个step梯度持续有限、epoch0 LR=9.045e-7；resume `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142300_j5_full_resume` 从epoch1继续且history=[0,1]；双卡DDP `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142500_j5_full_ddp` 两rank聚合、checkpoint、Pareto和5/64 stress一致。
- [x] 全预算J5正式训练完成：tmux `v4_j5_full_142455`，GPU0/5，输出 `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455`，日志 `/nfs-medical3/zyh/v4/phase6/logs/j5_full_budget_20260722_142455.log`，dead-status=0、EXIT_CODE=0。run_metadata确认world_size=2、5 epochs、每轮20,000 train/4,000 validation、test_used=false、唯一训练组为layer4；五个immutable checkpoint和每轮stress/Pareto产物齐全。
- [x] 全预算J5五轮Pixel macro/micro为0.737634/0.811187、0.739860/0.812560、0.740677/0.813524、0.740956/0.813305、0.741328/0.813801。best与Pareto均选epoch4；相对J3固定reference提升+0.008825/+0.009263，相对旧小预算J5正式validation提升约+0.007203/+0.006744。epoch4 Region/Unprompted/Boundary为0.818895/0.804936/0.314241，冲突197/4000=4.925%，stress恰197条评估occurrence。
- [x] J10最终配置 `configs/phase6_joint_j10_parent_context_full_budget.yaml` 已生成，Dice reference严格写入全预算J5 epoch4的0.7413279258440644/0.813800707215175；从 `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth` 起步，只训练Phase4 adapter、LR=2.5e-5，0.001双Dice非劣与至少一项提升规则已启用。
- [x] J10门禁完成：零等价/首步梯度 `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_153729_gate_zero_grad` 确认初始parent token max-abs=0、gate=0，且唯一非零训练组为198,145参数的parent_context；32/64多步BF16 `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_154000_j10_full_bf16` 的16 step梯度持续有限；resume `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_154100_j10_full_resume` 从epoch1接续；双卡DDP `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_154200_j10_full_ddp` 完成两rank聚合、checkpoint和stress set，全部EXIT_CODE=0。
- [x] J10全预算正式训练完成：tmux `j10_full_20260722_154210`，GPU0/5，输出 `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_154210`，日志 `/nfs-medical3/zyh/v4/phase6/logs/j10_full_budget_20260722_154210.log`，dead-status=0、EXIT_CODE=0。run_metadata确认world_size=2、5 epochs、每轮20,000 train/4,000 validation、test_used=false，且唯一训练组为parent_context；五个checkpoint与逐轮stress set齐全。
- [x] J10自动选择epoch1：Pixel macro/micro=0.7415117378/0.8128239369，相对全预算J5 epoch4为+0.000183812/-0.000976770；macro正增益、micro回落未超过冻结的0.001非劣容差，因此通过自动晋升规则。Boundary=0.3143838、冲突197/4000=4.925%均保持；Region/Unprompted macro=0.8157187/0.8057814，仍作为soft warning。epoch2–4因micro回落超过0.001不合格。
- [x] J5/J10正式paired validation完成：双卡固定4000 sampled occurrences（3839个唯一episode index、161个重复采样occurrence），审计目录 `/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260722_170259`，EXIT_CODE=0、`split=val/test_used=false`；逐episode TP/FP/FN/Dice位于 `episode_metrics.parquet`，J5/J10冲突均为197/4000=4.925%，两者geometry mismatch均为5256 slots，确认Phase4没有改变冻结geometry。
- [x] 10,000次paired episode bootstrap报告 `/nfs-medical3/zyh/v4/phase6/evaluation/paired_validation_20260722_170259/paired_report.json`：J10−J5 macro=+0.000183812，95% CI [-0.000274307,+0.000636968]，改善概率0.7872、可证明0.001非劣；micro=-0.000976770，95% CI [-0.001419147,-0.000559261]，改善概率0、非劣概率0.5458，不能证明0.001非劣。点估计规则通过，但联合CI非劣规则失败。
- [x] 最终决定：Dice优先下不以J10覆盖J5正式主路，继续保留全预算J5 epoch4；J10 epoch1及完整Phase4实现作为研究候选保留，不删除、不宣称稳定Dice提升，也不读取已封存test。

## 统一验收和 checkpoint 选择规则

每一阶段必须按以下顺序：

1. 静态数据契约与 tensor shape 检查。
2. 代码级单元测试。
3. 一个代表 episode 的同路径单例。
4. 多步 AMP smoke，覆盖 optimizer、validation 和 checkpoint。
5. DDP smoke，确认所有 rank 有 batch/梯度且聚合指标正确。
6. checkpoint/resume smoke，确认从下一 epoch 继续。
7. 正式 validation-only 比较；模型方案和 checkpoint 规则冻结前不读取 test。

默认 checkpoint 门槛（Dice 主导）：

- 硬门槛：Pixel macro Dice 不低于 0.72（test 解封前最终冻结值；替代早期 0.7383 规则）。
- 硬门槛：Pixel micro Dice 不低于动态 online 基线 0.8017 - 0.003 = 0.7987。
- 软目标：Boundary F1≥0.2867、Region macro Dice≥0.84、Unprompted region Dice≥0.81、原始冲突/abstention rate≤7%；未达只 warning，不否决 Dice 合格 checkpoint。
- point/small/large 和 12 类必须分开报告。
- 冲突 episode 必须进入 stress set，推理必须 abstain，不允许静默输出 mask。

保留满足两项 Pixel Dice 硬门槛的 Pareto checkpoints，best 以当前配置的候选 Dice reference 为参照，首先最大化 `max(Pixel macro gain, Pixel micro gain)`；任一 Pixel Dice 提升即可晋升，另一项只需保持硬门槛。其余指标不丢失，但作为软诊断。

## 待办清单

- [x] 完成 Phase2→3→5→6 fine-only 正式联调。
- [x] 独立汇总 4000 validation episodes 指标。
- [x] 完成 48 张 point/small/large/hard-case 可视化。
- [x] 完成 J1 在线 target 和动态 prompt 重映射。
- [x] 完成 J1 单例、smoke、DDP、resume 门禁。
- [ ] 运行 J2 防遗忘稳定化联调。
- [x] 评估 J3 逐步解冻。
- [x] 运行 J4 fine-only vs parent-context 消融。
- [x] 运行 J5 有限端到端微调。
- [x] 完成 J3 vs J5 独立 4000-episode validation-only 最终对照。
- [x] 完成 J6 全局像素阈值校准与正式4000 validation。
- [x] 完成 J7 layer4+decoder 联合适配小规模、DDP、resume与停止门禁。
- [x] 完成 J8 逐episode Dice权重定标与停止门禁。
- [x] 完成 J9 训练episode覆盖度扩大、双卡三轮与停止门禁。
- [x] 冻结最终方案：J5 epoch1、Pixel threshold=0.5、macro/micro门槛=0.72/0.7987、冲突推理abstain。
- [x] 最终候选冻结清单：/nfs-medical3/zyh/v4/phase6/final_selection/final_candidate_20260722_131830/final_candidate.json；包含checkpoint/config/正式validation summary/stress set的SHA-256，状态为`frozen_pre_test`，`test_evaluated=false`。
- [x] 冻结最终方案后仅读取一次 test；J5 macro/micro=0.739237/0.812773，正式通过。
- [ ] 完成全预算J5→J10 Phase4非劣接入；当前test不再读取。

## 变更记录

- 2026-07-21：创建联调方案和状态台账；记录已完成 fine-only 联调、独立 validation 结果、可视化驱动阻塞和 J1–J5 路线。J1 进入实现。
- 2026-07-21：J1 初版单例发现区域质心直接采样 assignment 会跳到相邻 slot；重映射契约修正为 online centroid 最近邻前向 + straight-through soft geometry 梯度。
- 2026-07-21：4000-episode mismatch 审计确认 49 条 cache/online 数值差异中，5 条 prompted 差异最高 purity=0.5003，3 条表面 purity=1 的差异仅含 1–2 个有效像素；高纯度稳定性门槛补充最小有效面积 1024 pixels，全部差异继续保存在 mismatch_details.parquet。
- 2026-07-21：J1 单元、单例、BF16 smoke、resume、双卡 DDP 和正式 4000-episode 可视化审计全部完成。正式审计发现旧 joint epoch24 有 280 个 online positive/negative slot conflicts；J2 进入实现，优先加入 prompt separation 和 Phase5 teacher 防遗忘约束。
- 2026-07-21：J2 核心损失、独立 initial-joint checkpoint 加载和 stabilization 配置完成；单例 20260721_223431 通过，新增损失、梯度、validation、checkpoint 均有限。J2 仍处于进行中，尚未启动正式训练。
- 2026-07-22：J2 prompt conflict 原始计数与 episode rate 接入 train、overall、12 类和 point/small/large validation。首次完整采样暴露 online region majority 可形成单类 episode；修正 J2 region auxiliary loss 的单类契约并新增显式可评估计数，Phase5 严格契约保持不变。
- 2026-07-22：三轮多类别/多尺寸单卡 BF16 smoke 20260722_002515 完成，EXIT_CODE=0。训练 separation/conflict 有波动，但固定 360 validation 的 conflict rate 三轮恒为 6.3889%，门禁失败；DDP、resume 和正式训练继续暂停。
- 2026-07-22：同集 checkpoint geometry 诊断确认训练前 conflict rate 为 5.8333%，J2 实际恶化至 6.3889%；soft overlap 与 centroid margin 基本不动。新增 J2b hard conflict margin 独立配置并完成单轮同预算 targeted 对照，validation 仍为 6.3889%，因此 J2b 被否决，未启动三轮、DDP、resume 或正式训练。
- 2026-07-22：完成冲突索引定位、分量梯度归因和 margin weight 1/5/10 overfit；weight5/10 可真实解除单 episode hard conflict。weight10 多 episode epoch0 将固定 validation 降至20/360，但 resume 后 epoch1/2反弹至23/360；集合诊断确认新增两个此前安全的冲突。cached Phase2 safe-geometry anchor 单轮为22/360并被否决。下一步改用冻结旧 joint epoch24 geometry teacher，正式训练继续暂停。
- 2026-07-22：实现冻结旧 joint epoch24 Phase2 geometry teacher 并共享 frozen backbone feature；冲突/安全单例和梯度路径通过。centroid/area teacher anchor 与 prompt Voronoi-gap teacher anchor 的两次256/360单轮门禁均为23/360，未优于训练前21，均被否决。停止 anchor 权重扫描、续 epoch、DDP 和正式训练；下一步改做冻结 Phase2 geometry 的 decoder-only 控制实验。
- 2026-07-22：decoder-only 单例、固定256/360三轮控制、同集旧 joint 基线和resume全部完成。三轮冲突稳定为21/360；epoch0相对旧 joint 同集基线小幅改善Pixel/Region/Unprompted，仅Boundary下降0.00024，成为当前J2候选。继续epoch收益不一致；正式训练仍暂停，下一步实现显式Pareto硬门槛后再做双卡DDP。
- 2026-07-22：固化 decoder-only 冻结 geometry 主路，新增仅训练过滤冲突 episode 的独立对照，validation 保留原始冲突率，并将冲突 episode 按 epoch 持久化为 stress set。256/360 对照排除23个训练冲突，validation 仍为21/360，stress set 与之严格一致。推理新增冲突 abstain 响应，不返回 mask 并明确提示调整 prompt。Phase6 25项测试通过；正式训练仍暂停。
- 2026-07-22：实现五指标硬门槛+非支配前沿 checkpoint 策略，Phase6 28项测试通过。代表性真实路径与显式 GPU 0,5 双卡 DDP smoke 均 EXIT_CODE=0；DDP 原始冲突5/64与两 rank stress shards 合并严格一致。该小样本未过 Boundary/冲突率硬门槛，因此产生 `no_eligible_checkpoint`，正式训练仍暂停。
- 2026-07-22：根据用户明确的最终 Dice 优先级，checkpoint 改为 Pixel macro/micro 硬门槛与主排序，Region/Unprompted/Boundary/冲突率改为不阻断的 soft targets。Phase6 29项测试通过；固定256/360实跑 20260722_022144 产生 Dice 合格 epoch0 best，Pixel macro/micro为0.74487/0.80316，stress set与21/360原始冲突一致，EXIT_CODE=0。下一步进入冻结 geometry 的 J3 Phase3 cell fusion/attention 小规模 Dice 对照。
- 2026-07-22：J3 Phase3 cell+decoder 小规模 Dice 对照完成。独立1e-5 cell LR、冻结geometry/Phase5的梯度门禁与最小smoke通过；256/360单轮使Pixel macro小幅上升0.000128，但micro下降0.000068。按双Dice不退化规则停止续epoch，decoder-only仍为主路，test未读取。
- 2026-07-22：J3 局部 Phase5+decoder 对照完成。实现matcher/task projection/Set Encoder最后层精确解冻与独立冻结teacher；梯度smoke通过。256/360单轮使Pixel micro +0.000546，但macro -0.000212。后续按用户确认的“任一 Pixel Dice 提升即可晋升”规则恢复至三轮，最佳 epoch1 的 micro 提升 +0.001343，macro 仍过硬底线，因此纳入候选；诊断项只产生 soft warnings，test未读取。
- 2026-07-22：J3 局部 Phase5+decoder 双卡 DDP 门禁通过，两 rank 的 decoder/prompt 梯度、分布式 validation 聚合、stress shards 和 checkpoint 产物均正常。候选 checkpoint 仍以固定360 validation 的 epoch1 为准，小样本 DDP Dice 不用于排名；test未读取。
- 2026-07-22：J4 parent-context 对照完成。零门控 adapter 从 J3 最佳起步，只训练两级父上下文路径；单例、多步 BF16、resume 和双卡 DDP 门禁全部通过。固定256/360三轮仅epoch0取得Pixel macro +0.000032，micro -0.000040且仍过底线，按用户规则纳入候选；后两轮双Dice下降，停止续训，test未读取。
- 2026-07-22：J5 有限端到端微调完成。仅解冻ResNet50 layer4，1e-6 LR，其他模块和BN运行统计冻结；单例梯度、多步BF16、resume和双卡DDP门禁全部通过。正确五轮cosine计划的最佳epoch1使Pixel macro/micro相对J3同时提升+0.002296/+0.004664，成为当前主候选；后三轮回退，不延长到8 epochs，test未读取。
- 2026-07-22：J3 vs J5 正式 4000-episode validation-only 对照完成。J5 的 Pixel macro/micro 相对 J3 均提升，但 macro=0.734125 低于0.7383硬底线，所以不晋升。评估仅使用validation，原始冲突率7.10%与stress set 284行均保留，test未读取。同时修正显式joint baseline被误当cache reference的审计门禁，失败与成功产物均保留。
- 2026-07-22：J6 单一全局像素阈值校准完成。单例、双卡64-episode DDP smoke 和正式4000 validation均EXIT_CODE=0。J5最优阈值0.475的macro/micro=0.734266/0.806286，macro仅比0.5提升0.000141，仍未过0.7383硬底线；停止阈值扫描，test未读取。
- 2026-07-22：J7 layer4+decoder 联合适配完成。两组保守LR下的单例、16-step BF16、双卡DDP和三轮resume门禁全部通过。固定360最佳epoch0相对J5仅macro +0.000494、micro -0.000905，不足以弥补正式4000的macro缺口，因此不进入正式4000，test未读取。
- 2026-07-22：J8 逐episode Dice loss权重定标完成。相对J7仅将pixel Dice权重1→2，单例梯度和16-step BF16门禁通过；固定360 epoch0双Dice相对J7均轻微下降，因此立即停止续epoch、DDP和正式4000，test未读取。
- 2026-07-22：J9 训练episode覆盖度扩大完成。保持J7模型/loss/LR不变，双卡将每轮train从256扩到2048；三轮最佳macro相对J5仅+0.000039，micro轻微下降，未达预设的正式4000晋升门槛。停止更大预算，test未读取。
- 2026-07-22：用户在test解封前取消早期0.7383 macro硬底线，将最终Pixel macro/micro门槛冻结为0.72/0.7987。J5正式4000 validation 的0.734125/0.807057通过双门槛，且相对J3同时提升，因此冻结J5 epoch1与0.5像素阈值为最终候选；历史运行仍按其当时规则记载。冲突validation不删除、stress set保留、推理显式abstain。
- 2026-07-22：最终候选已在读取test指标前写入不可变冻结清单 final_candidate_20260722_131830，记录完整输入哈希与选择检查。test前置审计发现现有Phase4/5产物仅含train/val；评估器与token构建器已增加显式test入口且禁止回退，单patch无Dice前处理门禁通过，开始生成独立test token/label/eligibility。
- 2026-07-22：独立test token/label/eligibility全部完成并通过行数、split与退出码契约；评估入口先用validation单例验证。随后按冻结方案唯一一次运行4000-episode test，J5 Pixel macro/micro=0.739237/0.812773，较J3分别+0.001258/+0.002847并通过0.72/0.7987门槛。冲突233/4000原样报告并与233行stress set一致；结果已封存且禁止test后选模或调参。
- 2026-07-22：用户确认将近似中性的Phase4重新作为完整六模块主路候选。新方案为先运行5×20,000/4,000的全预算J5控制，再从其最佳checkpoint零门控接入Phase4、仅训练adapter。非劣门槛冻结为macro/micro相对J5各不低于-0.001且至少一项提升，绝对0.72/0.7987门槛不变；最终用固定validation逐episode配对报告决定，已封存test不再运行。
- 2026-07-22：全预算J5单例、多步BF16、resume和双卡DDP门禁全部通过，Phase6共41项测试；正式双卡5轮训练 `phase6_joint_pixel_20260722_142455` 已启动，唯一训练组为layer4，20,000/4,000 episode预算和test隔离均由run_metadata确认。J10在该run产生最佳validation checkpoint前不创建带旧reference的最终配置。
- 2026-07-22：全预算J5双卡五轮完成，EXIT_CODE=0。Pixel macro/micro从epoch0的0.737634/0.811187稳步到epoch4的0.741328/0.813801，best指向epoch4；Boundary升至0.314241，冲突降至197/4000。J10配置已只用该validation结果写入reference，Phase4门禁与训练继续保持test隔离。
- 2026-07-22：J10 Phase4 parent-context 的零等价、仅adapter梯度、16-step BF16、resume和双卡DDP门禁全部通过。正式双卡五轮训练 `phase6_joint_pixel_20260722_154210` 已启动，只训练198,145个parent-context参数，20,000/4,000 episode预算与test隔离均由run_metadata确认；预计含训练后paired validation与台账封存共80–100分钟。
- 2026-07-22：J10正式双卡五轮完成，EXIT_CODE=0。自动规则选择epoch1，Pixel macro相对J5增加0.000184、micro下降0.000977，恰好满足“至少一项提升、另一项最多下降0.001”的冻结非劣规则；冲突率不变为4.925%，Boundary微升。该结果支持Phase4近似无损接入，但提升幅度很小，最终覆盖J5前仍执行固定validation逐episode paired CI，不读取已封存test。
- 2026-07-22：完成J5 epoch4与J10 epoch1的正式4000-occurrence paired validation和10,000次bootstrap。macro差值95% CI跨0；micro差值CI为[-0.001419,-0.000559]且下界越过-0.001非劣界，因此联合CI门禁失败。最终正式主路保留J5，Phase4/J10作为完整六模块研究候选保留；冲突、stress set、冻结geometry与test隔离契约均通过。
