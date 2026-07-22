
# 二、导师理想方法的完整架构

建议将方法拆成六个核心模块。

---

# 模块一：多尺度 WSI 金字塔输入

导师强调 WSI 本身就是金字塔，因此模型输入不应该只有一张固定 10x 图。

可以使用三个物理尺度，例如：

```text
Fine scale：10x
Middle scale：5x
Coarse scale：2.5x
```

或者在 10x 上使用不同感受野：

```text
Fine：1024 × 1024，对应细胞和小腺体
Middle：2048 × 2048，对应局部组织
Coarse：4096 × 4096，对应大片癌变和组织布局
```

训练时使用空间对齐的多尺度 crop：

```text
同一个中心坐标
├── Fine crop
├── Middle crop
└── Coarse crop
```

这样不同尺度描述的是同一片组织，只是视野不同。

## 每个尺度主要负责什么

| 尺度     | 主要信息                |
| ------ | ------------------- |
| Fine   | 细胞形态、腺体边缘、小坏死灶、细小间质 |
| Middle | 腺体结构、肿瘤间质、组织组合关系    |
| Coarse | 癌变区域、肌层、大片脂肪和组织布局   |

需要注意：

> 不是规定某个类别只能用某个尺度，而是给模型提供多个尺度，让模型根据 prompt 自己决定依赖哪些尺度。

---

# 模块二：深度区域化——同时学习边界和区域特征

这是导师方法与当前方法最本质的区别。

当前流程是：

```text
SLIC 先生成 superpixel
→ 再对每个 superpixel 聚合特征
```

理想流程是：

```text
Deep Region Encoder
→ 同时输出 soft region assignment
→ 同时输出 region embedding
```

## 模型输出

对于每个尺度，模型输出两部分：

### 1. 区域分配图

记作：

[
A_s \in \mathbb{R}^{H_s \times W_s \times K_s}
]

其中每个像素对多个区域有一个 soft assignment：

[
A_s(x,y,k)
]

表示像素 ((x,y)) 属于第 (k) 个区域的概率。

开始时可以是软分配，推理时再转成硬区域。

### 2. 区域特征

根据分配图对像素特征进行加权聚合：

[
r_{s,k}
=======

\frac{\sum_{x,y} A_s(x,y,k)F_s(x,y)}
{\sum_{x,y} A_s(x,y,k)}
]

这样得到每个尺度的 region token：

```text
R_fine
R_middle
R_coarse
```

## 为什么使用 soft region 而不是直接预测硬 superpixel

因为 soft assignment 可以反向传播。

最终分割损失可以推动模型改变区域边界：

```text
如果一个区域总是横跨 target 和 background
→ segmentation loss 会变大
→ 模型逐渐调整 assignment
→ 让区域边界更贴近语义边界
```

这才是真正的“为了下游任务学习 superpixel”。

---

# 模块三：将细胞级信息注入区域 token

导师特别强调：

> 判断癌变区域不能只看宏观颜色，还要看里面有没有癌细胞。

因此模型需要独立的 cell branch。

你的现有细胞特征可以继续利用，不需要废弃。

## 输入

每个细胞有：

```text
cell coordinate
cell embedding
cell regression feature
cell projection feature
cell type probability（如果有）
```

## Cell-to-region aggregation

对每个区域找到其中的细胞，聚合成区域级 cell context：

[
c_{s,k}
=======

\operatorname{CellAggregator}
\left(
{z_i \mid p_i \in region_{s,k}}
\right)
]

聚合方式可以采用：

* attention pooling；
* Set Transformer；
* 基于区域 token 的 cross-attention；
* mean + max + density statistics。

更推荐 prompt-independent 的 attention pooling：

[
\alpha_i
========

\operatorname{softmax}
\left(
q_{s,k}^{T}Wz_i
\right)
]

[
c_{s,k}
=======

\sum_i \alpha_i z_i
]

最终区域 token 为：

[
\tilde r_{s,k}
==============

\operatorname{Fusion}
\left(
r_{s,k}, c_{s,k},
\text{cell density},
\text{cell statistics}
\right)
]

这比直接 concat 更好，因为模型会学习：

* 哪些细胞对这个区域重要；
* 宏观区域应该关注哪些微观模式；
* 低细胞区域中“细胞少”本身也是一种证据。

---

# 模块四：跨尺度区域交互

这是导师说的：

> 让细微特征约束上一层级，把底层细节往宏观上面传。

首先需要建立尺度之间的父子关系。

例如一个 coarse region 覆盖多个 fine regions：

```text
Coarse region C1
├── Fine region F1
├── Fine region F2
├── Fine region F3
└── Fine region F4
```

根据空间重叠构建层级图：

```text
fine node → middle node → coarse node
```

## 方法一：层级图消息传递

细粒度信息向上聚合：

[
m_{c}
=====

\operatorname{Attention}
\left(
q=r_c,,
K=R_{\text{children}},,
V=R_{\text{children}}
\right)
]

[
r_c'
====

r_c + m_c
]

宏观上下文也可以向下传：

[
r_f'
====

r_f
+
\operatorname{Attention}
\left(
q=r_f,,
K=R_{\text{parent}},,
V=R_{\text{parent}}
\right)
]

这样：

* 大尺度 token 获得细胞和局部组织证据；
* 小尺度 token 获得整个区域的组织上下文。

## 方法二：Cross-scale Transformer

将三种尺度 token 放入一个层级 Transformer：

```text
Fine tokens
Middle tokens
Coarse tokens
Scale embedding
Position embedding
Parent-child relation embedding
```

但不建议做全连接 attention，因为 WSI region 数量太多。

应使用稀疏连接：

* 同尺度邻接区域；
* fine-middle 父子边；
* middle-coarse 父子边；
* 少量空间 KNN 边。

这样计算量更可控，也符合组织结构。

---

# 模块五：Prompt encoder

用户每次标注仍然是一个 target-vs-rest 二分类任务。

模型不需要提前知道类别名称。

输入可以包括：

```text
positive point / box / scribble / region
negative point / box / scribble / region
```

## Prompt token 怎么构造

假设用户正例覆盖一组区域：

[
P^+ = {r_i^+}
]

负例覆盖：

[
P^- = {r_j^-}
]

不要再只做简单均值 prototype。

可以使用一个 Set Encoder：

[
q^+
===

\operatorname{SetEncoder}(P^+)
]

[
q^-
===

\operatorname{SetEncoder}(P^-)
]

再组合成 task token：

[
q
=

\operatorname{MLP}
\left(
[q^+, q^-, q^+-q^-]
\right)
]

这样模型能处理：

* 一个或多个正例；
* 一个或多个负例；
* prompt 内部多样性；
* 不同 prompt 的相对差异。

## Prompt 与多尺度区域的匹配

每个区域不再只做 cosine：

[
s_i = \cos(r_i,q^+)-\lambda\cos(r_i,q^-)
]

而是通过 cross-attention 或 matching network 联合建模：

[
h_i
===

\operatorname{CrossAttention}
\left(
q=r_i,,
K=[q^+,q^-],,
V=[q^+,q^-]
\right)
]

最终输入：

[
u_i=[r_i,h_i,s_i,\text{geometry}_i,\text{scale}_i]
]

让模型学习不同情况下应该怎样解释相似度。

---

# 模块六：Context-aware mask decoder

最后输出每个 region 属于当前 target 的概率：

[
p_i = P(y_i=1 \mid \text{image},\text{prompt})
]

但它不是独立节点分类，而应同时利用：

* 当前区域 token；
* prompt token；
* 邻接区域；
* 父子尺度区域；
* 初始相似度；
* 区域边界信息。

可使用稀疏图 Transformer 或层级 GNN。

输出后根据 region assignment 投影回像素：

[
M(x,y)
======

\sum_k A(x,y,k)p_k
]

最终得到连续概率图。

再用固定的 0.5 阈值或简单校准即可，不再依赖 P90、类别面积先验或 classwise threshold。

真正的目标是：

> 让模型直接学习概率，而不是学习应该取 top 10% 还是 top 20%。

---

# 三、完整模型的训练目标

导师理想方法不能只靠一个 Dice loss，需要多个互相约束的目标。

可以将总损失写为：

[
\mathcal{L}
===========

\lambda_{seg}\mathcal{L}*{seg}
+
\lambda*{reg}\mathcal{L}*{region}
+
\lambda*{bd}\mathcal{L}*{boundary}
+
\lambda*{cs}\mathcal{L}*{cross-scale}
+
\lambda*{ret}\mathcal{L}*{retrieval}
+
\lambda*{comp}\mathcal{L}_{compact}
]

---

## 1. Prompt segmentation loss

每次训练采样一个 target class 和一组 prompts，输出 target-vs-rest mask。

使用：

[
\mathcal{L}_{seg}
=================

\mathcal{L}*{BCE}
+
\lambda_d\mathcal{L}*{Dice}
]

类别极不平衡时可以使用 focal BCE。

这是最终主任务。

---

## 2. Region purity loss

模型生成的区域应尽量语义纯净。

对于区域 (k)，根据 GT 计算其中各类别比例，鼓励一个区域不要混合多个类别。

可以通过像素标签分布熵实现：

[
\mathcal{L}_{purity}
====================

\sum_k
H
\left(
P(y\mid region_k)
\right)
]

熵越小，区域越纯。

但需要防止模型把每个像素都拆成独立区域，所以还要加紧凑性和平衡约束。

---

## 3. Boundary alignment loss

让区域边界贴近：

* tissue GT boundary；
* 图像边缘；
* 腺体或细胞密度变化边缘。

可以监督一个 boundary head：

[
\mathcal{L}_{boundary}
======================

\operatorname{BCE}
(B_{pred},B_{gt})
]

其中 (B_{gt}) 可以由组织 GT mask 计算。

不过组织标注边界可能粗糙，因此不能完全依赖它。

更合理的是组合：

```text
组织标签边界
+
图像梯度边界
+
细胞密度变化
```

---

## 4. Compactness loss

避免区域变得非常零碎。

例如约束每个像素与区域中心距离：

[
\mathcal{L}_{compact}
=====================

\sum_{x,y,k}
A(x,y,k)
\left|
p_{xy}-\mu_k
\right|^2
]

它类似传统 superpixel 的空间紧凑约束。

---

## 5. Cross-scale consistency loss

同一位置在不同尺度下对 target 的判断应基本一致：

[
\mathcal{L}_{cs}
================

D
\left(
P_{\text{fine}},
\operatorname{Upsample}(P_{\text{middle}})
\right)
+
D
\left(
P_{\text{middle}},
\operatorname{Upsample}(P_{\text{coarse}})
\right)
]

但不能要求完全相同，因为尺度负责的信息不同。

可以只对高置信度区域做一致性约束，边界区域降低权重。

---

## 6. Retrieval/ranking loss

在生成最终 mask 之外，还要保证 target region 的表示更接近正例、远离负例：

[
\mathcal{L}_{rank}
==================

\max
\left(
0,
m-s(r^+,q)+s(r^-,q)
\right)
]

或者使用 supervised contrastive loss。

这个损失直接优化 region embedding，让后续 prompt retrieval 真正可分。

---

# 四、数据集应该如何重新准备

现在没有时间限制，建议不要只围绕 300 个 prompt 建数据。

300 个优质 prompt 可以作为人工测试集，但完整模型的训练应基于组织 GT 自动生成大量 episodic tasks。

---

# 1. 基础数据组成

每个 WSI 至少需要：

```text
HE WSI pyramid
组织级 GT mask
细胞位置
细胞特征
患者 / 医院 / 切片元数据
```

可选但有帮助：

```text
细胞类型标签
腺体或细胞实例 mask
坏死、黏液等特殊区域标注
```

---

# 2. 训练单位不要直接用整张 WSI

训练单位应为多尺度对齐 patch：

```text
一个中心点
├── fine crop
├── middle crop
├── coarse crop
├── tissue GT
└── cell tokens
```

例如以 10x 对应区域为基础：

```text
Fine：1024 × 1024
Middle：覆盖 Fine 的 2 倍视野
Coarse：覆盖 Fine 的 4 倍视野
```

实际输入都可以 resize 到统一网络尺寸，但要保留物理尺度 embedding。

---

# 3. 需要多少数据

下面是经验性的项目规划，不是绝对门槛。

## 最低可验证规模

```text
30–50 张 WSI
5,000–10,000 个有效多尺度 crop
主要组织类别均有覆盖
```

可以验证模型能否正常工作，但泛化结论有限。

## 推荐规模

```text
100–300 张 WSI
30,000–100,000 个多尺度 crop
多个患者，最好多个中心
每个主要组织类至少数千个有效区域
```

适合较完整地训练和评估。

## 更理想规模

```text
300+ WSI
多医院 / 多扫描仪 / 多癌种
100,000+ 多尺度 crop
```

可以支持跨中心、跨癌种和 unseen-class 实验。

如果你目前只有约 20 张 WSI，也不是完全不能开始，但更适合：

* 证明架构可行；
* 做方法消融；
* 使用强预训练 backbone；
* 冻结大部分编码器；
* 不适合声称强泛化。

---

# 4. Prompt episode 自动生成

一个训练 episode 定义为：

```text
image crop
target class
positive prompts
negative prompts
target binary mask
```

每个 crop 可以产生很多 episode。

例如一张 crop 中有：

```text
tumor
stroma
lymphocyte
mucus
```

就可以构造四个 target-vs-rest 任务。

## Prompt 类型必须多样化

应包含：

* point；
* small box；
* large box；
* scribble；
* partial region；
* 多正例；
* hard negative；
* 带少量污染的 noisy prompt。

## Prompt 质量分层

建议显式划分：

```text
clean：purity > 0.9
mild noisy：0.7–0.9
hard noisy：0.5–0.7
```

这样模型不仅学习理想 prompt，也能适应真实用户操作。

---

# 五、推荐的训练顺序

不要一开始就把所有模块端到端联合训练，会非常不稳定。

建议分五个阶段。

---

# 阶段 0：彻底分析当前失败

在开发新模型之前，先回答：

1. 三个可学习模块的 train loss 是否下降？
2. train Dice 是否提升、test 不提升，还是训练集也不提升？
3. 不同尺度下的 oracle best Dice 分别是多少？
4. 当前 superpixel purity 上限是多少？
5. 使用 GT region label 训练线性分类器的上限是多少？
6. 若使用像素级 GT 直接选择最佳 superpixel，理论最高 Dice 是多少？

尤其要计算一个指标：

## Superpixel Oracle Dice

对于每个 superpixel，根据 GT 多数类别直接赋值，再计算 Dice。

这相当于：

> 假设分类完全正确，当前 superpixel 边界最多能达到多少 Dice？

如果 oracle Dice 只有 0.82，那么你的真实方法很难达到 0.9。

如果 small / middle / large oracle 都低，说明必须学习更好的区域边界。

如果 oracle 很高但实际 Dice 低，说明主要问题在 region representation 和 prompt matching。

这个实验决定后续重点。

---

# 阶段 1：训练深度区域化模块

先暂时不做 prompt segmentation。

目标是让模型输出的区域：

* 紧凑；
* 边界贴合；
* 语义纯净；
* 多尺度稳定。

可以使用：

```text
SLIC pseudo-label initialization
+
boundary loss
+
reconstruction loss
+
compactness loss
+
semantic purity loss
```

一开始让深度模型模仿 SLIC，训练稳定后逐步降低 SLIC 蒸馏权重，让最终分割任务重新调整区域边界。

---

# 阶段 2：训练区域表示

冻结或半冻结 regionization，训练：

```text
image region feature
+
cell aggregation
+
texture/context
```

使用：

* region classification；
* supervised contrastive；
* target-vs-rest episodic retrieval；
* cross-scale consistency。

这一步的目标是让 region token 在嵌入空间中真正具有组织可分性。

要持续评估：

```text
linear probe
kNN
retrieval mAP
supervised upper bound
类间 / 类内距离
```

---

# 阶段 3：训练跨尺度交互

加入层级图或稀疏 Transformer。

先只训练多尺度 region classification，验证：

```text
fine only
middle only
coarse only
naive concat
cross-scale interaction
```

确认跨尺度模块本身有收益后，再进入 prompt 任务。

否则一上来接 mask，无法判断是哪个模块失败。

---

# 阶段 4：Prompt-conditioned episodic training

每个 batch 随机采样：

```text
一个 WSI / crop
一个 target class
1–N 个正例 prompt
0–N 个负例 prompt
```

模型输出二值 mask。

训练时让类别不断变化：

```text
本 batch target = tumor
下个 batch target = mucus
再下个 batch target = stroma
```

模型学到的是：

> 根据当前视觉 prompt 定义目标，而不是固定输出某个类别。

这样推理时不需要针对新任务重新训练。

---

# 阶段 5：端到端联合微调

最后才联合调整：

```text
region boundary
region feature
cell aggregation
cross-scale interaction
prompt encoder
mask decoder
```

建议：

* backbone 使用较小学习率；
* 新模块使用较大学习率；
* regionization 的 compactness loss 保留；
* 逐步解冻，而不是全部同时训练。

---

# 六、模型第一版不要做得过于庞大

虽然现在没有时间限制，也不建议第一版直接做一个巨大 Transformer。

推荐第一版：

```text
预训练病理 backbone
+
三尺度特征金字塔
+
可微 soft region assignment
+
cell attention pooling
+
稀疏 hierarchical GNN
+
prompt set encoder
+
binary node decoder
```

这已经完整覆盖导师的核心思想。

第一版暂时不要做：

* 大型视觉语言模型；
* 文本类别 prompt；
* 全 WSI dense Transformer；
* 过于复杂的生成式 mask decoder；
* 从零训练 pathology foundation model。

重点先证明：

> 学习的 region boundary + 学习的 region feature + 跨尺度交互，确实优于固定 SLIC + 后处理学习模块。

---

# 七、关键实验应该怎么设计

完整实验需要围绕“每个升级是否真的有效”展开。

## 主对比

```text
Patch retrieval
Single-scale SLIC retrieval
Multi-scale SLIC retrieval
SLIC + learned postprocessor
Deep regionization only
Deep regionization + cell
Deep regionization + cross-scale
完整模型
```

## 必须做的消融

| 消融                               | 验证内容             |
| -------------------------------- | ---------------- |
| 去掉 cell branch                   | 微观细胞证据是否有效       |
| 去掉 cross-scale                   | 跨尺度信息是否有效        |
| 固定 SLIC 替代 learned region        | 深度区域化是否有效        |
| 去掉 negative prompt               | 负例交互是否有效         |
| mean prototype 替代 prompt encoder | prompt 建模是否有效    |
| 单尺度替代多尺度                         | 多尺度是否必要          |
| 独立尺度融合替代层级交互                     | 真正的跨尺度交互是否必要     |
| 去掉 purity/compactness loss       | region loss 是否有效 |

---

# 八、评价指标除了 Dice 还要增加什么

主指标仍然可以是：

```text
Dice
mIoU
mAP
BF1
Precision
Recall
```

但新方法需要加入一些更能解释架构的指标。

## 1. Region purity

每个区域内部 GT 类别的一致程度。

## 2. Boundary adherence

学习区域边界和组织 GT 边界的贴合程度。

## 3. Oracle region Dice

验证区域划分的理论上限。

## 4. Prompt efficiency

比较：

```text
1 positive
1 positive + 1 negative
1 positive + 3 negatives
4 positives
```

需要多少 prompt 才达到某个 Dice。

## 5. Unseen-WSI / unseen-patient

训练和测试必须按患者或 WSI 分开。

## 6. Unseen-class episode

更强的实验是：

```text
训练时不使用某些组织类作为 prompt task
测试时再用视觉 prompt 查找这些类
```

如果还能工作，才能支持较强的 class-agnostic / novel-target generalization 叙事。

## 7. Cross-site generalization

如果能获得不同医院数据，这是非常有价值的实验。

---

# 九、你的论文定位会发生什么变化

当前方法可以说：

> frozen representation 下的 training-free tissue-level in-context inference。

一旦训练完整模型，就不能再说“组织分割无需训练”。

更准确的定位应是：

> 一个经过 episodic training 的通用视觉提示组织分割模型，在推理时能够根据新的正负视觉 prompt 完成 target-vs-rest segmentation，不需要针对每个类别重新训练。

也就是：

```text
训练一次通用模型
测试时用户给新 prompt
无需 per-class fine-tuning
```

这仍然是很强的设定，只是不再是严格 training-free。

---

# 十、建议你现在先做的三项决定性实验

在真正开始完整模型前，先做三个实验，它们会决定模型重点。

## 实验一：Superpixel Oracle Upper Bound

对每个尺度，使用 GT 给 superpixel 赋最优标签。

输出：

```text
small oracle Dice
medium oracle Dice
large oracle Dice
multi-scale oracle Dice
```

判断边界是不是主要瓶颈。

---

## 实验二：GT Prompt + Supervised Region Classifier

使用非常干净的 GT prompt，再训练一个监督 region classifier。

如果监督分类仍然不高，说明 token 表征不足。

如果监督分类很高、prompt retrieval 低，说明 prompt matching 不足。

---

## 实验三：像素特征与区域特征对比

在相同 backbone 下比较：

```text
pixel/patch decoder
SLIC region decoder
learned soft region decoder
```

判断 superpixel 这种中间表示是否真的带来收益。

---

# 十一、最终实施路线

整个项目可以分成四个里程碑。

## 里程碑 1：诊断上限

完成：

* oracle region Dice；
* region purity；
* supervised region upper bound；
* 三个失败模块的 train/test 分析。

目标：明确问题究竟在边界、特征还是 prompt matching。

---

## 里程碑 2：学习型多尺度 region encoder

完成：

* 多尺度输入；
* soft region assignment；
* boundary/compactness/purity loss；
* region token 输出。

目标：学习区域的 oracle Dice 和 purity 明显优于 SLIC。

---

## 里程碑 3：Cell-aware cross-scale representation

完成：

* cell-to-region aggregation；
* hierarchical region graph；
* fine-to-coarse 和 coarse-to-fine 交互。

目标：region classification 和 retrieval mAP 明显提升。

---

## 里程碑 4：通用 prompt-conditioned segmentation

完成：

* positive/negative prompt set encoder；
* episodic target-vs-rest training；
* context-aware mask decoder；
* unseen-WSI、unseen-class 和 prompt-efficiency 实验。

目标：形成完整导师理想方法。

---

# 最重要的结论

你现在不应该再继续优化：

```text
P90
classwise threshold
scale selector
score fusion
独立后处理 Transformer
```

这些实验已经说明，仅在固定分数后面增加学习模块，收益有限。

下一步真正要做的是：

> **把学习发生的位置往前移。**

具体就是：

```text
固定 SLIC
→ 改为可学习的多尺度区域划分

手工拼接 token
→ 改为 image-cell 联合区域表示

独立尺度结果
→ 改为层级跨尺度信息交互

静态 cosine matching
→ 改为经过 episodic training 的 prompt-conditioned matching

阈值转 mask
→ 改为直接预测区域概率并投影成 mask
```

一句话概括导师的最终目标：

> **训练一个以 WSI 金字塔和正负视觉 prompt 为输入、能够联合学习多尺度组织区域边界、细胞感知区域表征、跨尺度上下文和 target-vs-rest mask 的通用交互式组织分割模型。**
