
所有的新建代码编写到/home/zhaoyh/CellAtlas/benchmarks/v3  并且在脚本开头用中文注释写明作用。脚本需按照阶段分文件夹保存。
所有的输出信息输出到/nfs-medical3/zyh/v3 
每一阶段验证通过后，在每个二级标题后添加"已完成标记"，并且输出一个详细的报告在/home/zhaoyh/CellAtlas/benchmarks/v3 下的每个阶段文件夹下。同时，需要在第一阶段结束后，创建一个skill，以复用技能，后续每个阶段完成，在这个skill上继续补充。
所有新创建的文件夹以2607xx(年月日)xxxx(时间)_(文件夹名称) 命名。



总体目标：先完成：

> 多尺度 superpixel + prompt scale selection + prompt-conditioned mask decoder / graph refiner。

---

# Agent 执行总方案：Multi-scale Prompt-aware Superpixel Tissue Segmentation

## 0. 总目标

基于当前已经验证有效的 PRET-superpixel 主线，升级为导师建议的多尺度交互式组织分割框架。

核心思想：

```text
不是推翻当前流程；
而是在当前 prompt retrieval 主干上，
用可学习模块替代手工尺度选择、阈值校准、score-to-mask 转换。
```

最终目标方法：

```text
HE WSI / 10x 图像
    ↓
多尺度 superpixel / region token
    ↓
用户 positive / negative prompt
    ↓
Prompt Scale Selector 选择合适尺度
    ↓
Prompt-conditioned Mask Decoder / Graph Refiner
    ↓
Refined probability map
    ↓
Final mask
```

---

# 1. 实验目录 已完成标记

新建总目录：

```bash
/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner
```

建议结构：

```bash
/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/
  pret_superpixel/
    multiscale_tokens/
    prompt_tasks/
    scale_selection/
    mask_decoder/
    graph_refiner/
    evaluations/
    visualizations/
    reports/
```

不要覆盖已有目录：

```bash
/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v2_full10x_texture
```

---

# 2. 输入数据 已完成标记

优先复用已有数据：

```bash
/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/manifests/main_20.csv
```

已有资源包括：

```text
1. HE WSI path
2. 10x GT tissue mask
3. cell raw / reg / proj feature
4. cell coordinates
5. 1024 patch raw feature
6. 当前 full10x auto-physical superpixel 结果
7. 当前 image_cell_reg / texture_cellstats token
8. 当前 300 条左右优质 prompt
```

需要确认并汇总：

```text
image_id
he_path
gt_mask_path
cell_feature_path
cell_coord_path
existing_superpixel_path
prompt_csv_path
```

输出一个统一 manifest：

```bash
pret_superpixel/data_manifest_v3.csv
```

---

# 3. 阶段 A：构建多尺度 superpixel / region token 已完成标记

## 3.1 目标 已完成标记

为每张 WSI 生成三套 superpixel：

```text
small
medium
large
```

当前已完成版本不再复用旧 medium 结果，而是对 small / medium / large 三个尺度全部重新运行 SLIC。
SLIC 的 tissue mask 使用低倍率宽松策略 `lowmag_loose`：

```text
低倍生成宽松 tissue mask
中倍/各尺度只在 mask 内做 SLIC superpixel
低置信度区域额外保存，后续用于高倍复核
```

建议尺度：

```text
small  = 0.5 × base diameter
medium = 1.0 × base diameter
large  = 2.0 × base diameter
```

已完成的全量命令：

```bash
conda run -n aligner python benchmarks/v3/phase_a/build_multiscale_tokens.py \
  --generation_mode slic \
  --tissue_mask_mode lowmag_loose \
  --workers 12 \
  --force
```

全量输出验证通过：

```text
images: 20
scales: small / medium / large
image_scale_outputs: 60
validation: passed
```

---

## 3.2 每个尺度输出 已完成标记

对每个 image_id、每个 scale 输出：

```bash
multiscale_tokens/<image_id>/<scale>/
  he_10x_rgb.npy
  he_tissue_mask.npy
  he_tissue_high_conf.npy
  he_tissue_low_conf.npy
  superpixels.npy
  superpixels.csv
  adjacency.npy
  tokens_image_cell_reg.npy
  tokens_image_cell_reg_texture_cellstats.npy
  gt_label_stats.csv
  validation.json
```

其中 `scale` 为：

```text
small
medium
large
```

低倍 mask 相关文件说明：

```text
he_tissue_mask.npy       # lowmag_loose 宽松组织 mask，用于限制 SLIC 区域
he_tissue_high_conf.npy  # 低倍强证据组织区域
he_tissue_low_conf.npy   # 宽松 mask 内的低置信区域，后续高倍复核候选
```

---

## 3.3 每个 superpixel 需要保存的字段 已完成标记

`superpixels.csv` 至少包含：

```text
segment_id
area
center_x
center_y
bbox_x0
bbox_y0
bbox_x1
bbox_y1
cell_count
cell_density
gt_majority_label
gt_target_fraction
gt_purity
valid_fraction
```

---

## 3.4 token 类型 已完成标记

至少生成两种 token：

```text
base token:
image_cell_reg_cellw0p5

enhanced token:
image_cell_reg_texture_cellstats
```

主方法默认先用：

```text
image_cell_reg_cellw0p5
```

enhanced token 作为 ablation。

---

## 3.5 tissue mask 修复记录 已完成标记

已完成多轮 tissue mask 修复验证：

```text
1. HSV / RGB 颜色阈值检查
2. RGB + Sobel edge density 边缘纹理检查
3. 边缘闭合 / 浅色空腔填洞检查
4. 低倍率宽松 tissue mask + 低置信区域高倍复核方案
```

关键结论：

```text
中倍像素级颜色/边缘阈值能补一部分浅脂肪泡，但不稳定；
区域级闭合能提高 fat coverage，但容易吞外部白底；
低倍率宽松 mask 最稳定，能利用整体组织轮廓覆盖脂肪区域。
```

14 张脂肪问题图验证结果：

```text
lowmag_loose_mask:
  mean fat coverage: 93.4%
  mean background: 25.4%
  fat90_image_count: 11/14

lowmag_high_conf:
  mean fat coverage: 90.7%
  mean background: 19.8%
  fat90_image_count: 11/14
```

相关可视化输出：

```bash
/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/visualizations/phase_a_lowmag_mask_refine_check/
```

---

## 3.6 阶段 A 最终验证 已完成标记

最终验证文件：

```bash
/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/reports/phase_a_validation.json
```

最终报告：

```bash
/home/zhaoyh/CellAtlas/benchmarks/v3/phase_a/report.md
```

验证摘要：

```text
passed: true
generation_mode: slic
tissue_mask_mode: lowmag_loose
images: 20
image_scale_outputs: 60
scale_counts:
  small: 20
  medium: 20
  large: 20
```

---

# 4. 阶段 B：构建 prompt-task 数据集 已完成标记

## 4.1 数据单位 已完成标记

每条训练样本不是一张图，而是一次 prompt 分割任务：

```text
query = (image_id, target_class, positive_prompts, negative_prompts)
```

每个 query 的目标是：

```text
target-vs-rest binary segmentation
```

允许不同 query 的 mask 重叠。

---

## 4.2 Phase B 重设计 已完成标记

Phase B 不再沿用旧 prompt CSV 作为训练/评估数据。正式 prompt task 全部从 Phase A 当前
`lowmag_loose + slic` 生成的新 multiscale superpixel 中重新采样，prompt 的基础粒度是
`medium` superpixel。

旧文件：

```bash
/nfs-medical3/zyh/pret_eval_prompt_class_fix_20260706_gt_pure_prompts/prompts.csv
```

仅作为审计参考，不写入正式 `all_prompt_tasks.csv`。如需保留旧 prompt，仅输出：

```bash
prompt_tasks/legacy_prompt_audit.csv
```

并标记：

```text
usage_status = not_used
```

---

## 4.3 输入与采样规则 已完成标记

输入只使用：

```text
data_manifest_v3.csv
multiscale_tokens/<image_id>/medium/superpixels.csv
multiscale_tokens/<image_id>/medium/superpixels.npy
multiscale_tokens/<image_id>/medium/tokens_image_cell_reg.npy
GT mask 统计字段
```

`small` / `large` 不参与 Phase B prompt 采样，只在 Phase C retrieval 评分时参与。

采样规则：

```text
clean positive:
  gt_majority_label == target_class
  gt_purity >= 0.8
  valid_fraction >= 0.8

noisy positive:
  gt_majority_label == target_class
  0.5 <= gt_purity < 0.8
  valid_fraction >= 0.5

hard negative:
  gt_majority_label != target_class
  valid_fraction >= 0.5
  优先选择与 target clean prototype cosine 相似度高的非 target segment
```

每个有效 `(image_id, target_class)` 最多采样：

```text
clean 10
noisy 5
hard-negative 10
```

query 结构：

```
每个 query 至少 1 个 positive prompt
hard-negative query 配 1-3 个 negative prompts
同时保留 positive-only query 和 positive+negative query
```

prompt box 使用新 superpixel 的 `bbox_x0/y0/x1/y1`，坐标系为 Phase A 10x / GT mask 坐标系。
class id 保持数字 id，不做 class name 映射。

---

## 4.4 输出与验证 已完成标记

脚本：

```bash
/home/zhaoyh/CellAtlas/benchmarks/v3/phase_b/build_prompt_tasks.py
```

输出：

```bash
prompt_tasks/auto_prompt_tasks.csv
prompt_tasks/all_prompt_tasks.csv
prompt_tasks/prompt_task_summary.csv
prompt_tasks/legacy_prompt_audit.csv
reports/phase_b_validation.json
visualizations/phase_b_prompt_task_samples/
/home/zhaoyh/CellAtlas/benchmarks/v3/phase_b/report.md
```

不生成 `expert_prompt_tasks.csv` 作为正式数据。

最终验证摘要：

```text
passed: true
total_queries: 3711
unique_query_ids: 3711
images: 20
classes: 12
clean queries: 1293
noisy queries: 918
hard-negative queries: 1500
positive-only queries: 2211
positive+negative queries: 1500
legacy prompt rows audited as not_used: 1161
sample visualizations: 36
```

验证内容：

```text
重新计算 prompt_purity 并写入输出字段
hard negative 的 negative_gt_majority_label != target_class
每个 query 至少一个 positive prompt
输出类别覆盖表：每个 class 的 clean/noisy/hard-negative 数量
旧 prompt CSV 不进入正式 Phase B 输出
Phase B 不训练模型，不跑 retrieval
```

报告：

```bash
/home/zhaoyh/CellAtlas/benchmarks/v3/phase_b/report.md
```

验证 JSON：

```bash
/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/reports/phase_b_validation.json
```

---

# 5. 阶段 C：为每个 query 在三种尺度上跑 baseline retrieval 已完成标记

## 5.1 对每个 query、每个 scale 计算 score 已完成标记

对于每条 query：

```text
scale ∈ {small, medium, large}
```

计算：

```text
score_pos(x) = max_i cos(x, q_pos_i)
score_neg(x) = max_j cos(x, q_neg_j)
score(x) = score_pos(x) - lambda * score_neg(x)
```

默认：

```text
lambda = 0.5
```

如果没有 negative：

```text
score(x) = score_pos(x)
```

---

## 5.2 需要保存的 query-scale 结果 已完成标记

输出：

```bash
evaluations/query_scale_scores/<query_id>_<scale>.npz
```

包含：

```text
segment_ids
score_pos
score_neg
score_final
rank_percentile
gt_soft_label_per_superpixel
gt_hard_label_per_superpixel
area
center_xy
```

---

## 5.3 每个尺度计算指标 已完成标记

每个 query-scale 计算：

```text
mAP
AUROC
P@top5
P@top10
Dice_p90
Dice_p80
Dice_otsu
Dice_gmm
Dice_global_toparea
Dice_classwise_toparea
BestDice
BestArea
mIoU
BF1@5
BF1@10
PredArea
GTArea
Precision
Recall
FP_area
FN_area
```

输出：

```bash
evaluations/multiscale_baseline_metrics.csv
```

脚本：

```bash
/home/zhaoyh/CellAtlas/benchmarks/v3/phase_c/run_multiscale_baseline.py
```

最终运行命令：

```bash
conda run -n aligner python benchmarks/v3/phase_c/run_multiscale_baseline.py --workers 4
```

实现说明：

```text
prompt 来自 Phase B medium superpixel。
medium 尺度优先使用原始 prompt segment id。
small / large 尺度使用 prompt bbox 在对应尺度选择 center 命中 segment，必要时回退到 bbox overlap。
GT hard/soft label 使用 Phase A 已标准化 superpixels.csv 中的 gt_majority_label / gt_purity / valid_fraction 字段。
gt_soft_label_per_superpixel = target-majority segment 的 gt_purity * valid_fraction，否则为 0。
```

最终验证摘要：

```text
passed: true
prompt queries: 3711
query-scale rows: 11133
score npz files: 11133
ok metric rows: 11126
degenerate_target rows: 7
scale_counts:
  small: 3711
  medium: 3711
  large: 3704 ok + 7 degenerate_target
```

报告：

```bash
/home/zhaoyh/CellAtlas/benchmarks/v3/phase_c/report.md
```

验证 JSON：

```bash
/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/reports/phase_c_validation.json
```

像素级正式评估已完成：将预测的完整 superpixel 回填到图像，与原始 GT 像素直接计算指标。

```text
overall PixelDice_classwise_toparea: 0.4014
overall Pixel_mIoU: 0.2701
overall PixelBestDice: 0.4925
small / medium / large PixelDice: 0.4277 / 0.4065 / 0.3700
```

正式结果使用：

```bash
evaluations/multiscale_pixel_metrics.csv
evaluations/pixel_class_scale_summary.csv
```

`GT-presence` 只作为 coverage diagnostic；后续阶段 D/E/F 的 Dice/mIoU 主结果使用 pixel-level 口径。

---

# 6. 阶段 D：训练 Prompt Scale Selector 已完成标记

## 6.1 目标 已完成标记

训练一个模型自动选择当前 prompt 最适合的尺度：

```text
small / medium / large
```

导师想法对应：

> 用户画得小，用小尺度；画得大，用大尺度。

---

## 6.2 训练标签 已完成标记

对每个 query，三种尺度都已经有指标。

定义 best scale：

```text
best_scale = argmax Dice_classwise_toparea
```

或者：

```text
best_scale = argmax Dice_prompt_adaptive
```

第一版用：

```text
BestDice 或 classwise_toparea Dice
```

生成：

```bash
scale_selection/scale_selector_dataset.csv
```

字段：

```text
query_id
image_id
target_class
prompt_area
prompt_box_w
prompt_box_h
prompt_aspect_ratio
prompt_token_count_small
prompt_token_count_medium
prompt_token_count_large
prompt_purity
score_std_small
score_std_medium
score_std_large
score_gap_small
score_gap_medium
score_gap_large
best_scale
```

---

## 6.3 模型 已完成标记

先训练轻量模型：

```text
RandomForestClassifier
GradientBoostingClassifier
MLPClassifier
```

主结果先用表现最稳的。

不要直接上大 Transformer。

---

## 6.4 切分方式 已完成标记

必须 WSI-level split。

建议：

```text
5-fold by image_id
```

不要随机按 query split。

输出：

```bash
scale_selection/scale_selector_results.csv
scale_selection/scale_selector_model.pkl
```

指标：

```text
scale accuracy
macro F1
chosen-scale Dice
oracle-scale Dice
medium-only Dice
small-only Dice
large-only Dice
```

核心比较：

```text
medium-only
manual prompt-size rule
learned scale selector
oracle best scale
```

最终五折 WSI-level 结果（pixel-level 主指标）：

```text
fixed small PixelDice: 0.4284
fixed medium PixelDice: 0.4072
fixed large PixelDice: 0.3700
best learned selector: GBDT / formal pixel-Dice label
  PixelDice: 0.4293
  Pixel mIoU: 0.2933
  scale accuracy: 0.6261
  macro F1: 0.3286
```

selector 的提升相对 fixed small 只有 0.0009；它是一个已验证的基线模块，但不是当前主要性能瓶颈。正式模型和 OOF 结果位于：

```bash
scale_selection/scale_selector_model.pkl
scale_selection/scale_selector_oof_predictions.csv
scale_selection/scale_selector_results.csv
```

---

# 7. 阶段 E：训练 Prompt-conditioned Mask Decoder

## 7.1 目标

替代当前手工 threshold / calibration。

输入：

```text
superpixel token + prompt score + geometry + context feature
```

输出：

```text
每个 superpixel 属于 target 的概率
```

---

## 7.2 每个 node 的输入特征

对每个 query-scale-superpixel，构建 node feature：

```text
token_base
token_enhanced_optional
score_pos
score_neg
score_final
rank_percentile
area
x_norm
y_norm
cell_density
cell_count
texture_stats_optional
distance_to_positive_prompt
distance_to_negative_prompt
neighbor_mean_score
neighbor_max_score
neighbor_std_score
```

标签：

```text
soft_label = target_class_fraction_in_superpixel
hard_label = 1 if soft_label > 0.5 else 0
```

---

## 7.3 模型版本

### Version 1：MLP Decoder

```text
Input: node feature
Output: target probability
Loss: BCE + Dice loss
```

这是最稳 baseline。

### Version 2：Graph Refiner

基于 superpixel adjacency：

```text
node = superpixel
edge = adjacent superpixel
node feature = token + score + geometry
```

模型：

```text
GraphSAGE
GAT
GCN
```

输出：

```text
node probability
```

优先实现：

```text
GraphSAGE
```

因为更稳。

---

## 7.4 损失函数

使用组合损失：

```text
loss = BCEWithLogitsLoss(pos_weight) + lambda_dice * SoftDiceLoss
```

建议：

```text
lambda_dice = 1.0
```

类别不平衡时加入：

```text
pos_weight = negative_area / positive_area
```

---

## 7.5 训练协议

必须 WSI-level 5-fold。

每个 fold：

```text
train images: 16
val images: 2
test images: 2
```

训练数据：

```text
auto_prompt_tasks + train images 中的 expert prompts
```

测试时分别报告：

```text
automatic prompt setting
expert prompt setting
```

---

## 7.6 输出

```bash
mask_decoder/mlp_decoder_results.csv
graph_refiner/graphsage_refiner_results.csv
```

每个结果需要包含：

```text
fold
image_id
query_id
scale
model
Dice
mIoU
BF1@5
BF1@10
Precision
Recall
PredArea
GTArea
```

---

# 8. 阶段 F：多尺度融合实验

## 8.1 目标

验证导师说的多尺度是否有价值。

对每个 query，有三个尺度 score/probability：

```text
small
medium
large
```

测试以下融合策略：

### A. best single scale oracle

```text
三种尺度中 Dice 最高
```

只作为上限。

### B. prompt-size rule

根据 prompt 面积选择尺度。

### C. learned scale selector

用阶段 D 模型选择尺度。

### D. score-level fusion

```text
final_prob = w_s * prob_small + w_m * prob_medium + w_l * prob_large
```

权重由模型预测，或者先用简单 MLP。

### E. max fusion

```text
final_prob = max(prob_small, prob_medium, prob_large)
```

---

## 8.2 输出

```bash
evaluations/multiscale_fusion_results.csv
```

主表比较：

```text
single medium baseline
manual scale rule
learned scale selector
multi-scale score fusion
oracle best scale
```

---

# 9. 阶段 G：Expert prompt 上限评估

## 9.1 目标

验证你说的 300 条优质 prompt 是否可以稳定达到 Dice 0.8。

单独建立：

```text
Expert Prompt Benchmark
```

---

## 9.2 设置

对 300 条 expert prompt 跑：

```text
current baseline
1pos + 3 strict hard neg
prompt-adaptive calibration
classwise calibration
scale selector
mask decoder
graph refiner
multi-scale fusion
```

输出：

```bash
reports/EXPERT_PROMPT_BENCHMARK.md
```

重点指标：

```text
Dice
mIoU
BF1
Precision
Recall
per-class Dice
per-image Dice
```

如果 Dice 达到 0.8，要明确说明：

```text
expert prompt setting
high-quality prompt upper bound
```

不要和自动 prompt 混在一起比较。

---

# 10. 阶段 H：消融实验

至少做以下 ablation：

## 10.1 特征消融

```text
image only
cell only
image + cell
image + cell + texture/cellstats
```

## 10.2 Prompt 消融

```text
1pos
1pos + 1neg
1pos + 3neg
expert prompt
automatic prompt
clean prompt
noisy prompt
```

## 10.3 多尺度消融

```text
small only
medium only
large only
manual scale rule
learned scale selector
oracle scale
```

## 10.4 模型消融

```text
P90
prompt-adaptive calibration
classwise calibration
MLP decoder
GraphSAGE refiner
multi-scale fusion
```

输出：

```bash
reports/ABLATION_RESULTS.md
```

---

# 11. 阶段 I：可视化

每个主设置生成 best / median / worst cases。

至少包括：

```text
1. 当前 baseline
2. hard negative
3. learned scale selector
4. graph refiner
5. expert prompt high-Dice case
6. failure case
```

每个 case 输出：

```text
HE image
GT mask
positive prompt overlay
negative prompt overlay
small-scale score map
medium-scale score map
large-scale score map
selected scale
baseline mask
refined mask
error map TP/FP/FN
```

保存：

```bash
visualizations/
```

每个 case 保存：

```text
summary.json
```

---

# 12. 最终报告

输出主报告：

```bash
reports/AAAI_V3_MULTISCALE_REFINER_RESULTS.md
```

报告结构：

```markdown
# AAAI V3 Multi-scale Prompt-aware Superpixel Tissue Segmentation

## 1. Motivation
当前流程依赖手工尺度选择、阈值校准、score-to-mask。

## 2. Method Overview
多尺度 superpixel / region token
prompt scale selector
prompt-conditioned mask decoder / graph refiner
multi-scale fusion

## 3. Dataset
20 WSI
tissue GT
cell features
300 expert prompts
auto-generated prompt tasks
WSI-level split

## 4. Multi-scale Superpixel Baseline
small / medium / large 对比

## 5. Scale Selector Results
manual rule vs learned selector vs oracle

## 6. Mask Decoder / Graph Refiner Results
P90 / calibration / MLP / GNN 对比

## 7. Expert Prompt Benchmark
300 high-quality prompts 的上限结果

## 8. Ablation
feature / prompt / scale / model

## 9. Visualization
best / median / worst cases

## 10. Conclusion
```

---

# 13. 主结果表格式

最终主表建议如下：

| Method                 | Scale    | Prompt | Learnable? | Dice | mIoU | BF1@5 | mAP |
| ---------------------- | -------- | ------ | ---------: | ---: | ---: | ----: | --: |
| Current P90            | medium   | auto   |         No |  ... |  ... |   ... | ... |
| Current classwise      | medium   | auto   |         No |  ... |  ... |   ... | ... |
| 1pos+3neg              | medium   | auto   |         No |  ... |  ... |   ... | ... |
| Manual scale rule      | multi    | auto   |         No |  ... |  ... |   ... | ... |
| Learned scale selector | multi    | auto   |        Yes |  ... |  ... |   ... | ... |
| MLP mask decoder       | selected | auto   |        Yes |  ... |  ... |   ... |   - |
| Graph refiner          | selected | auto   |        Yes |  ... |  ... |   ... |   - |
| Graph refiner          | selected | expert |        Yes |  ... |  ... |   ... |   - |
| Oracle best scale      | multi    | auto   |     Oracle |  ... |  ... |   ... |   - |

---

# 14. 20 天执行时间表

## Day 1-2：数据整理

任务：

```text
整理 manifest
整理 expert_prompts.csv
生成 prompt_tasks
确认 WSI-level split
```

产出：

```text
data_manifest_v3.csv
all_prompt_tasks.csv
fold_split.csv
```

---

## Day 3-5：多尺度 superpixel/token

任务：

```text
生成 small / medium / large superpixels
生成 tokens
生成 adjacency
生成 GT label stats
```

产出：

```text
multiscale_tokens/
multiscale_validation.json
```

---

## Day 6-7：三尺度 baseline retrieval

任务：

```text
每个 query 在 small/medium/large 上跑 retrieval
计算指标
```

产出：

```text
multiscale_baseline_metrics.csv
```

---

## Day 8-9：Scale Selector

任务：

```text
构建 best-scale label
训练 RF/GBDT/MLP scale selector
和 manual rule / oracle 比较
```

产出：

```text
scale_selector_results.csv
```

---

## Day 10-12：MLP Mask Decoder

任务：

```text
构建 node dataset
训练 MLP
评估 automatic / expert prompts
```

产出：

```text
mlp_decoder_results.csv
```

---

## Day 13-15：Graph Refiner

任务：

```text
构建 superpixel graph dataset
训练 GraphSAGE / GAT
评估
```

产出：

```text
graph_refiner_results.csv
```

---

## Day 16：Multi-scale Fusion

任务：

```text
manual scale rule
learned selector
score fusion
oracle best scale
```

产出：

```text
multiscale_fusion_results.csv
```

---

## Day 17：Ablation

任务：

```text
feature ablation
prompt ablation
scale ablation
model ablation
```

产出：

```text
ABLATION_RESULTS.md
```

---

## Day 18：可视化

任务：

```text
best/median/worst
expert prompt case
failure case
```

产出：

```text
visualizations/
```

---

## Day 19-20：报告整理

任务：

```text
整理主表
整理图
写 AAAI_V3_MULTISCALE_REFINER_RESULTS.md
```

---

# 15. 验收标准

最低验收：

```text
1. 三尺度 superpixel/token 全部生成完成；
2. 每个 query 在 small/medium/large 上都有 baseline 指标；
3. learned scale selector 跑通；
4. MLP decoder 跑通；
5. GraphSAGE refiner 跑通；
6. expert prompt benchmark 跑通；
7. 输出完整报告和可视化。
```

理想指标目标：

```text
automatic prompt:
Dice >= 0.62
mIoU >= 0.49

expert prompt:
Dice >= 0.75
最好接近 0.80

learned scale selector:
优于 medium-only 和 manual rule

graph refiner:
优于 classwise / prompt-adaptive calibration
```

---

# 16. 注意事项

## 16.1 不要训练完整 deep superpixel generator

20 天内不要做：

```text
HE image → deep model → superpixel boundary + feature
```

这个作为 future work 或导师理想目标。

本轮只做：

```text
multi-scale SLIC superpixel
+
learned scale selector
+
learned mask decoder / graph refiner
```

---

## 16.2 不要随机按 prompt 切分

必须按 WSI split。

否则数据泄漏。

---

## 16.3 expert prompt 和 automatic prompt 分开报告

不要把 expert prompt 的 0.8 Dice 和 automatic prompt 混报。

报告中明确：

```text
automatic prompt setting
expert prompt setting
```

---

## 16.4 classwise calibration 要注明使用 class label

主结果里最好同时报告：

```text
不使用 class label 的结果
使用 class label 的 calibrated 结果
```

---

## 16.5 Graph Refiner 的输入不能只用 GT 派生特征

训练时可以用 GT 做 label。

但测试输入不能包含：

```text
gt_purity
gt_label
gt_area
```

只能用：

```text
token
score
geometry
prompt
adjacency
```

---

# 17. 最终论文故事

如果实验成功，方法可以描述为：

> 我们提出了一个多尺度 prompt-aware superpixel in-context tissue segmentation framework。它保留了 PRET-style prompt retrieval 的灵活性，同时通过 learned scale selection 和 prompt-conditioned graph refinement 替代手工尺度选择与阈值校准，从而更适合交互式病理组织标注。

核心贡献：

```text
1. Multi-scale superpixel region tokenization
2. Prompt-aware scale selection
3. Prompt-conditioned graph mask refinement
4. Expert prompt benchmark and interaction analysis
```

---

# 18. Agent 最后需要回答的三个问题

跑完后报告必须明确回答：

```text
Q1: 多尺度 small/medium/large 是否比单一 medium 更好？
Q2: learned scale selector 是否能替代手工尺度选择？
Q3: mask decoder / graph refiner 是否能替代手工 threshold calibration，并提升 Dice/mIoU/BF1？
```

如果这三个问题答案都是 yes，这一版就能作为 20 天内最现实的 AAAI 方法主线。
