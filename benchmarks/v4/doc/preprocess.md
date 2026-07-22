# 第一阶段：10x 数据工程与监督分割基线——Agent 执行任务书

> 本文件可以直接交给代码 Agent 执行。Agent 必须先检查当前仓库结构和已有实现，再将下面模块集成进现有工程；不要另起一个与现有代码重复的孤立项目。

## 1. 已知背景

- 当前约有 1000 张 HE WSI。
- 每张 WSI 对应一张 10x 分辨率的组织分割 GT 图。
- 最终目标是训练“多尺度 WSI + cell-aware region token + visual prompt”的通用 target-vs-rest 组织分割模型。
- 但本阶段只完成数据基础设施和一个可靠的 10x 多类别监督分割基线，不实现多尺度、可学习 regionization、cell branch、prompt encoder 或 mask decoder。
- 10x GT 是统一监督、输出和评价坐标系；后续阶段再添加 5x、2.5x 上下文。

## 2. 本阶段核心目标

按以下顺序完成，任何一步未通过验收都不得跳到后面：

1. 审计 WSI、GT 和元数据，确认配对关系、类别编码、10x 空间对齐以及未标注区域含义。
3. 建立不复制原始图像的可复现 10x patch 索引。
4. 编写 PyTorch Dataset、数据增强、监督分割模型、loss、训练和验证代码。
5. 先做 16 个 patch 的过拟合测试，证明坐标、标签、loss 和模型实现正确。
6. 过拟合测试通过后，运行 1 epoch smoke test。
7. smoke test 通过后才能启动完整 10x baseline 训练。

本阶段的科学问题只有一个：

> 在不经过 superpixel、prompt 和后处理的情况下，10x HE 对组织类别的监督分割上限有多高？

## 3. Agent 开始前必须执行的检查

1. 阅读仓库中的 `AGENTS.md`、README、环境配置和已有数据代码。
2. 使用 `rg --files` 和 `rg` 检查是否已经存在：
   - WSI reader；
   - GT reader；
   - patient split；
   - patch dataset；
   - segmentation model；
   - Dice/mIoU 实现；
   - DDP 和日志框架。
3. 优先复用v3中已有可靠实现，将代码复制到v4文件夹下。
4. 先输出简短的仓库检查结论和拟修改文件列表，再开始编辑。
5. 不执行破坏性 Git 操作，不覆盖用户已有实验结果。

## 4. 必须配置化的输入

创建或补充一个类似下面的配置文件。字段可以适配现有配置框架，但不得把路径、类别或倍率写死在 Python 中。

```yaml
project:
  name: phase1_10x_supervised_baseline
  seed: 20260714

data:
  wsi_root: /PATH/TO/WSI_ROOT
  gt_root: /PATH/TO/GT_ROOT
  metadata_csv: /PATH/TO/METADATA.csv
  manifest_path: data_manifests/wsi_gt_manifest.parquet
  split_path: data_manifests/patient_split.csv
  patch_index_path: data_manifests/patch_index_10x.parquet

  target_magnification: 10
  target_mpp: null
  patch_size: 512
  index_stride: 512
  ignore_index: 255
  min_valid_fraction: 0.50
  min_tissue_fraction: 0.30

  class_map:
    # 必须按真实 GT 编码填写，不能猜测
    # 0: background
    # 1: tumor
    # 2: stroma

split:
  train: 0.70
  val: 0.15
  test: 0.15
  group_key: patient_id
  stratify_keys: [center, dominant_class]

model:
  name: deeplabv3_resnet50
  num_classes: null
  pretrained: true

training:
  epochs: 50
  batch_size_per_gpu: 8
  num_workers: 8
  optimizer: adamw
  lr: 0.0001
  weight_decay: 0.0001
  amp: true
  grad_accum_steps: 1
  early_stop_patience: 10
  monitor: val_macro_dice
  loss:
    ce_weight: 0.5
    dice_weight: 1.0
    use_class_weights: true

output:
  root: runs/phase1_10x_baseline
```

如果仓库已有 Hydra、Lightning、MMEngine 或其他配置体系，融入已有体系，不要强制新建 YAML 解析器。

## 5. 数据处理任务

### 5.1 WSI—GT 配对和 manifest

实现数据审计脚本，例如：

```text
tools/audit_10x_dataset.py
tools/build_wsi_gt_manifest.py
```

每张 WSI 至少记录：

```text
wsi_id
patient_id
wsi_path
gt_path
center
scanner
level0_width
level0_height
base_mpp_x
base_mpp_y
objective_power
available_levels
level_downsamples
gt_width
gt_height
gt_dtype
gt_unique_values
inferred_gt_downsample_x
inferred_gt_downsample_y
alignment_status
split
```

要求：

1. WSI 和 GT 必须通过明确 ID 或 metadata 配对，不允许依靠两个目录排序后 `zip`。
2. 无匹配 WSI、重复 GT、多重候选和打不开的文件单独输出报告，不能静默跳过。
3. 如果缺少 `patient_id`，先报告风险；只有用户明确确认“一张 WSI 对应一个独立患者”后，才能临时使用 `wsi_id` 分组。
4. 记录所有 GT 唯一值和像素数量，确认 RGB/调色板颜色到类别 ID 的映射。
5. 明确三种区域：真实组织类别、明确背景、未标注/不确定区域。未标注区域必须映射到 `ignore_index`，不得默认当背景。

### 5.2 10x 空间对齐检查

不能只相信文件名中的“10x”。使用 WSI 元数据、GT 尺寸和物理分辨率检查：

```text
expected_downsample_x = level0_width / gt_width
expected_downsample_y = level0_height / gt_height
```

检查：

- X/Y 缩放是否近似一致；
- GT 是否存在裁边、padding、旋转、翻转或坐标偏移；
- GT 是否覆盖整张 WSI，还是仅覆盖 ROI；
- WSI 是否存在各向异性 MPP；
- 10x 图像读取时应使用哪个原始 level，或是否需要在相邻 level 上重采样。

随机选择至少 20 张 WSI，每张选择多个位置，生成 HE+GT 半透明叠加预览和类别边界预览。输出 contact sheet，供人工确认。

验收要求：

- 所有进入训练的数据都必须具有明确且可复现的坐标变换；
- 发现偏移时，不允许靠训练增强“吸收偏移”；必须修正配准或将异常 WSI 排除并报告；
- 图像使用双线性/区域插值，类别 mask 只能使用最近邻插值。


### 5.4 10x patch 索引

不要提前导出几十万张 PNG。构建只包含坐标和统计信息的 patch index，训练时动态读取 WSI 和 GT。

每个 patch 至少记录：

```text
patch_id
wsi_id
patient_id
split
x_10x
y_10x
width_10x
height_10x
x_level0
y_level0
target_mpp
valid_fraction
tissue_fraction
boundary_fraction
present_classes
class_pixel_counts
dominant_class
sampling_group
```

`sampling_group` 至少分为：

```text
class_interior
class_boundary
rare_class
low_cell_or_special_tissue
background_or_hard_negative
```

第一版默认使用 `512×512 @10x` 训练输入。索引 stride 可以设为 512；后续需要更大视野时只修改配置，不重写 Dataset。

采样器需要避免背景和大类支配训练。推荐每个 batch 近似：

```text
30% 类别内部
30% 组织边界
20% 稀有类别
10% 特殊/困难区域
10% 背景或困难负例
```

如果实际类别定义不支持某一组，根据数据统计调整，并在报告中说明。

## 6. 第一版训练代码

### 6.1 建议文件结构

根据现有仓库调整命名，但功能至少应拆分为：

```text
configs/phase1_10x_baseline.yaml
src/data/wsi_reader.py
src/data/gt_reader.py
src/data/segmentation_dataset.py
src/data/balanced_patch_sampler.py
src/models/segmentation_baseline.py
src/losses/segmentation_losses.py
src/metrics/segmentation_metrics.py
src/train_phase1_10x.py
src/eval_phase1_10x.py
tools/audit_10x_dataset.py
tools/build_wsi_gt_manifest.py
tools/build_10x_patch_index.py
tests/test_coordinate_mapping.py
tests/test_mask_decode.py
tests/test_dataset_sample.py
tests/test_loss_metrics.py
```

不要为了匹配该列表而破坏现有工程结构。

### 6.2 Dataset 返回内容

每个样本至少返回：

```python
{
    "image": FloatTensor[3, H, W],
    "mask": LongTensor[H, W],
    "valid_mask": BoolTensor[H, W],
    "wsi_id": str,
    "patch_id": str,
    "coords_10x": Tensor[4],
}
```

要求：

- `image` 正确归一化；
- `mask` 只能含合法类别 ID 和 `ignore_index`；
- 图像和 mask 的空间增强必须完全同步；
- 不允许对 mask 使用双线性插值；
- val/test 禁止随机增强；
- reader 应支持 worker 内延迟打开 WSI，避免在主进程提前打开后被 fork；
- 为每个 worker 设置随机种子；
- 对损坏 tile 给出明确错误和 WSI ID，不静默返回黑图。

### 6.3 数据增强

训练集第一版只使用安全增强：

- 水平/垂直翻转；
- 90 度旋转；
- 轻量 brightness/contrast/saturation/hue；
- 轻量 H&E stain augmentation（若仓库已有可靠实现）。

不要第一版加入强几何形变、随机缩放或会改变物理倍率的增强。

### 6.4 模型

第一版目标是可靠基线，不追求复杂创新。默认优先级：

1. 如果仓库已有成熟分割框架，使用现有框架；
2. 否则使用 `torchvision` 的 DeepLabV3-ResNet50；
3. 若已经安装并使用 `segmentation_models_pytorch`，可使用 ResNet50-UNet/FPN；
4. 不得为了第一阶段从零实现复杂 Transformer decoder。

模型输出：

```text
logits: [B, num_classes, H, W]
```

预训练权重必须可配置。离线环境找不到权重时给出清晰提示，允许指定本地 checkpoint；不要偷偷退回随机初始化。

### 6.5 Loss

默认：

```text
L = 0.5 * CrossEntropy(ignore_index=255)
  + 1.0 * multiclass Soft Dice
```

规则：

- Dice 只统计有效类别和有效像素；
- 默认不把 background 纳入 macro Dice，但同时单独报告 background；
- 类别权重只使用训练集统计计算；
- 对极稀有类别的权重需要截断，避免单个 batch 梯度爆炸；
- loss 每个组成项分别记录。

### 6.6 训练功能

训练脚本至少支持：

- 单 GPU 与 `torchrun` DDP；
- AMP；
- 梯度累积；
- AdamW；
- warmup + cosine decay 或已有稳定 scheduler；
- best/last checkpoint；
- 从 checkpoint 恢复 optimizer、scheduler、scaler 和 epoch；
- TensorBoard 或现有实验跟踪工具；
- 每轮保存 train/val loss、macro Dice、macro mIoU 和 per-class 指标；
- 固定随机种子和完整配置快照；
- checkpoint 中保存 class map、ignore index 和数据 split hash。

### 6.7 验证与指标

至少计算：

```text
macro Dice（不含background）
macro mIoU（不含background）
per-class Dice
per-class IoU
precision
recall
pixel confusion matrix
```

指标先在 patch 上累计 confusion，再按 WSI 汇总；最终报告同时给出：

```text
micro/pixel-weighted
patch-weighted
WSI-macro
class-macro
```

不能只报告所有像素混在一起的 micro Dice，否则大类别会掩盖小类别失败。

本阶段可以先在验证索引上评价，但必须实现支持滑窗拼接整张 10x GT 区域的推理接口。重叠区使用概率平均或平滑权重融合，禁止直接覆盖。

## 7. 必须完成的正确性测试

### 7.1 单元测试

至少覆盖：

1. level0 坐标、10x 坐标和 GT 像素坐标的往返映射；
2. RGB/palette GT 到类别 ID 的解析；
3. 最近邻缩放后类别集合不发生变化；
4. Dataset 图像和 mask 同步增强；
5. ignore 区域不进入 loss 和指标；
6. Dice/mIoU 在手工小矩阵上的结果正确；
7. Dataloader 多 worker 读取不会崩溃。

### 7.2 16-patch 过拟合测试

关闭随机增强，固定 16 个包含多个类别的 patch。

要求：

```text
训练loss持续下降
训练macro Dice达到0.95左右或非常接近1
预测mask与GT视觉一致
不存在恒定输出单一类别
```

如果不能过拟合，不允许启动全量训练。优先检查：

- GT 类别映射；
- 图像—GT 坐标；
- logits 尺寸；
- ignore 处理；
- Dice 实现；
- 学习率和预训练归一化。

### 7.3 1-epoch smoke test

过拟合通过后，在完整训练索引的小规模子集上运行 1 个 epoch，验证：

- DDP/AMP 正常；
- checkpoint 保存和恢复正常；
- validation 完整跑通；
- 日志和指标文件完整；
- 显存与吞吐可接受；
- 不出现 NaN/Inf。

## 8. 第一阶段验收标准

### 8.1 数据验收

- WSI—GT 成功配对率接近 100%；未配对项全部列出原因。
- 训练数据不存在未知 GT 值。
- 随机 20 张 WSI 的叠加预览经过人工确认。
- train/val/test 没有患者重叠。
- 每个主要类别在 train/val/test 中均有覆盖；不能覆盖时明确报告。
- split、manifest 和 patch index 重复生成具有一致结果。

### 8.2 代码验收

- 单元测试通过。
- 16-patch 过拟合 macro Dice 约 0.95 以上。
- 1-epoch smoke test 通过。
- 中断后可从 checkpoint 继续训练。
- 任意预测都能追溯到 wsi_id、patch_id 和 10x 坐标。

### 8.3 模型验收

完整 baseline 暂不设置不合理的绝对论文门槛，因为组织类别和 GT 质量尚未统计。采用以下诊断规则：

- train loss 与 validation loss 均呈合理下降；
- validation macro Dice 建议至少达到 0.60；若低于 0.50，停止扩展模型并做错误分析；
- 常见类别 Dice 应明显高于随机和多数类基线；
- train—validation macro Dice 差距长期超过 0.15 时检查过拟合和数据分布；
- 输出至少 20 个最好、20 个最差和 20 个随机验证 patch 的可视化；
- 按类别、中心、扫描仪、组织面积和边界比例分析失败模式。

这里的 0.60/0.50 只是工程诊断线，不是最终论文目标。最终是否通过，主要看数据对齐、过拟合测试和相对于多数类/简单模型的可靠表现。

## 9. 明确禁止事项

本阶段禁止：

- 使用 test 集选择 checkpoint、类别权重、阈值或增强策略；
- 按 patch 随机划分数据；
- 把未标注区域默认当背景；
- 对类别 mask 使用双线性插值；
- 只汇报 micro Dice；
- 在数据审计失败时继续训练；
- 一次性加入 5x/2.5x、cell、regionization、prompt 和后处理；
- 用 GT 生成 superpixel 或把 GT 信息泄露到图像预处理；
- 静默删除坏数据或未知类别。

## 10. 建议执行命令

根据仓库实际入口调整，但最终应提供等价命令：

```bash
python tools/audit_10x_dataset.py \
  --config configs/phase1_10x_baseline.yaml

python tools/build_wsi_gt_manifest.py \
  --config configs/phase1_10x_baseline.yaml

python tools/build_10x_patch_index.py \
  --config configs/phase1_10x_baseline.yaml

pytest -q tests/test_coordinate_mapping.py \
  tests/test_mask_decode.py \
  tests/test_dataset_sample.py \
  tests/test_loss_metrics.py

python src/train_phase1_10x.py \
  --config configs/phase1_10x_baseline.yaml \
  --overfit-num-patches 16

torchrun --standalone --nproc_per_node=6 src/train_phase1_10x.py \
  --config configs/phase1_10x_baseline.yaml \
  --max-epochs 1 \
  --run-name smoke_test

torchrun --standalone --nproc_per_node=6 src/train_phase1_10x.py \
  --config configs/phase1_10x_baseline.yaml \
  --run-name full_baseline
```

不要未经确认直接启动长时间全量训练。完成数据审计、叠加预览、单元测试、过拟合测试和 smoke test 后，先报告结果，再启动正式训练。

## 11. Agent 最终必须交付

1. 修改/新增的代码和配置。
2. `wsi_gt_manifest.parquet`、`patient_split.csv`、`patch_index_10x.parquet`。
3. 数据审计报告和异常 WSI 清单。
4. 至少 20 张 WSI 的 HE/GT 叠加预览。
5. 单元测试结果。
6. 16-patch 过拟合曲线、指标和预测图。
7. 1-epoch smoke test 的日志、显存、吞吐和 checkpoint 恢复结果。
8. 一份 `PHASE1_REPORT.md`，必须包含：
   - 数据配对率；
   - 患者和 WSI 划分；
   - 类别定义和像素分布；
   - GT 对齐结论；
   - 训练命令；
   - 单元测试和过拟合结果；
   - smoke test 结果；
   - 已知问题；
   - 是否满足启动全量 baseline 训练的条件。

## 12. 本阶段完成后的下一步

第一阶段完整 baseline 训练完成并分析后，第二阶段才比较：

```text
10x only
10x + 5x context
10x + 5x + 2.5x context
```

确认多尺度监督 upper bound 有效后，再进入 10x learned regionization。不要跳过这个顺序。
