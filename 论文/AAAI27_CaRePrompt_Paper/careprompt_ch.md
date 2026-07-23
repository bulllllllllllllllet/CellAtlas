# CaRePrompt：从细胞证据到层级区域提示的交互式全切片组织分割

**匿名投稿**

## 摘要

全切片图像（whole-slide images，WSIs）中的交互式组织分割要求模型根据稀疏视觉提示识别用户指定的组织，同时整合细胞、区域和全局组织尺度的信息。现有方法通常依赖固定图像块或人工构建的区域，这可能限制其适应复杂组织边界以及捕获不同尺度生物学上下文的能力。

我们提出 CaRePrompt，一种面向 WSI 交互式目标与其余区域二分类分割的细胞感知层级提示框架。CaRePrompt 从多尺度切片特征中学习自适应组织区域表征，融合细胞信息以增强区域级理解，并利用层级上下文推理连接局部形态与更广泛的组织结构。通过以正、负视觉提示为条件进行分割，CaRePrompt 使单一统一模型无需针对类别进行微调即可分割用户指定的组织。

我们在患者分离的 WSI 队列上，以情景式交互分割设置评估 CaRePrompt。实验结果表明，所提出的框架能够有效整合细胞和空间上下文，并在多种组织目标上实现准确的组织轮廓勾画。这些发现表明，细胞感知层级提示有潜力成为交互式计算病理学的一种通用框架。

## 引言

CaRePrompt 将交互式 WSI 组织分割从孤立像素或图像块上的提示匹配，转变为在细胞感知的学习式组织区域层级上进行提示条件推理。这一转变解决了用户指定目标后仍然存在的一项限制：模型仍须判断哪些局部结构支持该目标、这些结构如何组合成组织，以及最终边界在切片尺度上的位置。在覆盖 12 种组织目标以及点、小区域和大区域交互的 4,000 个患者分离留出测试情景上，单一 CaRePrompt 模型无需针对类别进行微调，即可获得 0.7392 的宏平均 Dice 和 0.8128 的微平均 Dice。这些结果首先确立了本文的核心贡献：当细胞、区域和切片上下文被纳入同一预测路径时，视觉提示可以定义一个可复用的目标与其余区域二分类组织任务。

相应的失败发生在从提示到掩膜的接口处。细粒度图像块可以保留细胞核和狭窄边界，却会遗漏解释这些结构所需的组织构造；粗粒度图像块可以捕获这种构造，却会平均掉区分视觉相似组织的细胞证据。固定图像块或预计算超像素在看到提示之前便确定了这一取舍。即使检索过程识别出相关区域，将相似度排序转换成掩膜还会引入第二个失败点：逐类别阈值、面积先验或最高百分位规则对预测组织的影响可能超过提示本身。CaRePrompt 将这两项决策都移入模型内部。可微区域分配确定预测单元，细胞注意力提供微观证据，稀疏父子边提供多尺度上下文，学习式解码器则将带符号的提示证据直接映射为校准后的区域概率。

已有工作解决了这一问题的重要组成部分，但尚未将它们耦合起来。SAM 在自然图像中建立了可提示分割和零样本迁移（Kirillov et al. 2023），MedSAM 将框提示分割适配到多种医学成像模态（Ma et al. 2024）。WSI-SAM 进一步为提示式组织病理学分割引入了多分辨率特征（Liu et al. 2024）。与此同时，HIPT 表明天然的 WSI 金字塔支持层级表征学习（Chen et al. 2022），CellViT 则证明基于 Transformer 的细胞表征能够跨组织病理学数据集迁移（Hörst et al. 2024）。这些进展分别提供了提示、多分辨率上下文或细胞表征。缺失的一步，是让它们在相同的学习式组织区域上发生交互：细胞证据应改变区域语义，层级结构应跨物理尺度连接这些区域，而正、负提示应共同定义二元组织查询，而不是选择一个固定类别头。

我们的贡献围绕这一缺失的计算过程展开，并分别配有明确的实验检验：

**自适应区域到像素预测。** 我们将可微组织区域与上下文感知的区域到像素解码器相结合，以软概率投影取代检索截断。在采用动态在线区域目标、包含 4,000 个情景的受控验证对比中，这项改变将像素级宏平均 Dice 从 0.7212 提升至 0.7316，并将边界 F1 从 0.2287 提升至 0.2878。

**细胞感知的带符号提示。** 我们引入细胞感知区域 token 和带符号集合编码器，以表征多个正、负提示，而不将每个集合缩减为固定的均值原型。包含 360 个情景的受控消融实验分别隔离了两种作用：相对于冻结几何结构的解码器基线，加入细胞分支使宏平均/微平均 Dice 分别变化 $+0.00013/-0.00007$，适配提示匹配器则使二者分别变化 $-0.00122/+0.00134$。这些方向不一致的结果将本文主张限定为互补证据，而非一致性提升。

**稀疏层级上下文。** 我们通过零门控稀疏层级连接对齐的细、中、粗粒度区域，使模型可以在初始化时不扰动预训练细粒度表征的情况下加入更广泛的上下文。在配对的 4,000 情景对比中，使用 10,000 次 bootstrap 重采样测得宏平均 Dice 变化为 $+0.00018$，微平均 Dice 变化为 $-0.00098$；后者未能确立预先设定的非劣效界限。因此，我们将层级交互报告为一项经过检验的架构贡献，并保留性能选择出的细粒度模型作为主要路径，而不宣称缺乏证据支持的普遍增益。

**情景式目标与其余区域二分类学习。** 我们跨不断变化的目标组织，以情景方式训练完整预测路径，并在患者分离的模型选择完成后仅评估一次。在冻结测试集上，最终模型相较于其前一个提示条件检查点，将宏平均/微平均 Dice 分别提高 $+0.00126/+0.00285$，在 12 种组织目标中的 7 种上取得提升，并保持统一的 0.5 推理阈值。该实验直接检验了预期使用场景：一个模型、异构视觉提示，并且推理时无需针对类别进行微调。

## 相关工作

### 交互式与可提示分割

交互式分割最初通过面向特定任务的空间线索编码用户意图。DEXTR 将四个极值点点击转换为额外的图像通道，用于类别无关的目标分割（Maninis et al. 2018）；FocalClick 则执行粗粒度目标定位和聚焦校正，根据正、负点击高效更新掩膜（Chen et al. 2022）。基础模型进一步扩展了这一交互接口：SAM 接受点、框和掩膜，在自然图像上执行可提示分割（Kirillov et al. 2023）；MedSAM 将框提示适配到大规模医学图像（Ma et al. 2024）。在病理学领域，WSI-SAM 将多分辨率特征纳入基于 SAM 的全切片分割模型（Liu et al. 2024）。这些方法建立了有效的提示到掩膜预测，但其交互主要在图像级或特征级解码器中完成。CaRePrompt 则使用提示查询学习式组织区域，而这些区域的语义由细胞信息和跨尺度上下文共同决定。

### 多尺度全切片表征

WSI 分析通常将局部特征提取与更广泛的上下文聚合分开处理。HistoSegNet 根据图像块监督、类别激活图和空间后处理生成组织类型图（Chan et al. 2019）。HIPT 将嵌套的 WSI 图像块组织到层级自监督 Transformer 中，以连接局部表征和切片级表征（Chen et al. 2022）。近期，Prov-GigaPath 利用长上下文切片编码器和大规模全切片预训练，对数万个图像块 token 进行建模（Xu et al. 2024）。这些方法表明物理尺度和切片上下文至关重要，但它们主要面向预定义组织标签或切片级终点学习表征。本文的设置有所不同：输出类别在交互时才被指定，因此多尺度信息必须根据当前正、负证据，被路由到稠密的目标与其余区域二分类掩膜中。

### 学习式区域与细胞上下文

基于区域的模型将稠密图像压缩为结构化预测单元。Superpixel Sampling Networks 使超像素分配可微，并允许下游损失学习面向任务的区域（Jampani et al. 2018）。计算病理学进一步将具有生物学意义的实体引入这一抽象。HoVer-Net 跨多种组织联合执行细胞核分割和分类（Graham et al. 2019），CellViT 学习可在组织病理学数据集之间迁移的 Transformer 细胞表征（Hörst et al. 2024）。HACT-Net 更进一步，将细胞图与组织图连接成用于乳腺癌分类的层级表征（Pati et al. 2022）。这些工作为自适应区域和细胞到组织结构提供了依据。CaRePrompt 将二者结合到一种不同的预测接口中：软组织区域接收细胞证据，在对齐的物理尺度之间交换稀疏上下文，并在区域概率投影回像素之前保持可被带符号提示直接寻址。

## 方法

### 问题定义与总体框架

令 $I$ 表示一张 H&E WSI，$\Omega$ 表示其在参考分辨率下的像素域。一次交互提供一组正提示 $\mathcal{P}^{+}$ 和负提示 $\mathcal{P}^{-}$，其中每个提示可以是点、框、涂画或局部区域。提示指定的是一个视觉概念，而不是预定义的语义标签。因此，CaRePrompt 学习如下目标与其余区域二分类映射：

$$
f_{\theta}(I,\mathcal{P}^{+},\mathcal{P}^{-}) = \hat{M},
\qquad \hat{M}\in[0,1]^{|\Omega|}.
\tag{1}
$$

其中，$\hat{M}$ 是当前交互所指示组织的概率掩膜。单一模型跨不断变化的目标类别进行训练，并且无需针对类别进行微调即可响应新的查询。

CaRePrompt 将切片表示为学习式组织区域的层级结构。空间对齐的视图首先在细、中、粗三个尺度上提供互补的形态信息。可微区域编码器将稠密特征转换为软区域，随后细胞分支将细胞证据注入其 token。稀疏父子交互将局部形态与更广泛的组织构造相连接。最后，带符号提示编码器定义目标概念，上下文感知解码器预测区域概率，并将其投影回像素网格。

### 多尺度深度区域化

对于每个采样位置，我们提取对齐视图 $\{I_s\}_{s\in\mathcal{S}}$，其中 $\mathcal{S}=\{f,m,c\}$ 分别表示细、中、粗尺度。这些视图中心相同，但覆盖逐渐扩大的物理视野。因此，细粒度视图保留细胞形态和狭窄边界，而较粗粒度视图则捕获腺体、组织组成和全局构造，同时不将任何组织类别固定到某一尺度。

在尺度 $s$ 上，图像编码器生成稠密特征图 $F_s\in\mathbb{R}^{H_s\times W_s\times d}$。区域分配头预测 $K_s$ 个软区域：

$$
A_s(x,k)=
\frac{\exp(a_s(x,k))}
{\sum_{j=1}^{K_s}\exp(a_s(x,j))}.
\tag{2}
$$

其中，$A_s(x,k)$ 表示像素 $x$ 对区域 $k$ 的隶属度。对应的区域 token 通过归一化软池化得到：

$$
r_{s,k}=
\frac{\sum_x A_s(x,k)F_s(x)}
{\sum_x A_s(x,k)+\varepsilon}.
\tag{3}
$$

与固定超像素不同，区域边界及其表征均接收来自下游任务的梯度。区域预训练以传统过分割作为弱几何监督的起点，同时使用边界、紧致度、平衡和语义纯度目标防止分配碎片化或坍缩。随后的情景式训练可以进一步将区域边界移动到有利于提示条件分割的结构上。

### 细胞感知区域表征

在 H&E 图像中，仅凭宏观外观可能产生歧义；颜色和纹理相似的区域可能包含不同的细胞群体。因此，我们将每个检测到的细胞 $i$ 与其坐标 $u_i$ 和嵌入 $z_i$ 关联。在 $u_i$ 处计算的软分配决定哪些区域可以接收该细胞的证据。对于区域 $(s,k)$，注意力池化定义为：

$$
e_{s,k,i} =
\frac{(W_q r_{s,k})^{\top}(W_k z_i)}{\sqrt{d}},
\tag{4}
$$

$$
\alpha_{s,k,i} =
\frac{A_s(u_i,k)\exp(e_{s,k,i})}
{\sum_j A_s(u_j,k)\exp(e_{s,k,j})+\varepsilon},
\tag{5}
$$

$$
c_{s,k} = \sum_i \alpha_{s,k,i}W_v z_i.
\tag{6}
$$

这一构造利用区域外观选择信息丰富的细胞，同时将细胞密度和汇总统计量保留为显式证据。学习式融合层将 $r_{s,k}$、$c_{s,k}$ 及这些统计量组合成细胞感知 token $\tilde{r}_{s,k}$。该表征还显式建模低细胞密度组织，而不是将缺少细胞视为信息缺失。

### 稀疏层级区域交互

由于多尺度视图在空间上完成配准，区域重叠会在相邻层级之间产生父子关系。我们保留稀疏的细到中以及中到粗连接，而不对所有 WSI 区域施加稠密注意力。对于父区域 $v$，子区域证据聚合为：

$$
m_v = \operatorname{Attn}\!\left(
\tilde{r}_v,
\{\tilde{r}_u\}_{u\in\mathcal{C}(v)}
\right).
\tag{7}
$$

其中，$\mathcal{C}(v)$ 包含与其发生空间重叠的子区域。信息首先从细粒度汇总至中粒度，再从中粒度汇总至粗粒度。所得父级和祖先级上下文通过门控残差适配器返回每个细粒度区域：

$$
h_{f,k}=\tilde{r}_{f,k}
+\gamma\,\operatorname{HAttn}\!\left(
\tilde{r}_{f,k},\tilde{r}_{\pi(k)},\tilde{r}_{\pi^2(k)}
\right).
\tag{8}
$$

其中，$\pi(k)$ 和 $\pi^2(k)$ 分别表示父区域和祖先区域。门 $\gamma$ 初始化为零，因此引入层级上下文时不会扰动预训练的细粒度表征。这种稀疏交换使细胞和边界证据能够约束大尺度组织区域，同时将更广泛的组织构造返回局部预测。

### 带符号的视觉提示编码

提示根据归一化几何位置映射到当前的学习式区域，因此随着软分配演化，其关联仍保持有效。令 $R^{+}$ 和 $R^{-}$ 分别为正提示和负提示覆盖的区域 token 集合。两个独立的置换不变集合编码器分别汇总两种提示极性：

$$
q^{+}=\operatorname{SetEnc}^{+}(R^{+}),\qquad
q^{-}=\operatorname{SetEnc}^{-}(R^{-}),
\tag{9}
$$

并形成任务 token：

$$
q=\operatorname{MLP}([q^{+},q^{-},q^{+}-q^{-}]).
\tag{10}
$$

该表征支持数量和类型可变的提示，并保留每个集合内部的差异。对于每个候选区域，对 $q^{+}$ 和 $q^{-}$ 的交叉注意力产生提示条件特征，而带符号相似度则提供显式的吸引—排斥线索。解码器输入为：

$$
u_k=[h_{f,k},q,h_k^{\mathrm{prm}},s_k^{+}-s_k^{-},g_k].
\tag{11}
$$

其中，$h_k^{\mathrm{prm}}$ 是经过交叉注意力处理的提示特征，$g_k$ 包含区域几何信息。训练期间，目标以及提示到区域的关联根据当前分配重新计算，而不是从已经过时的区域划分中继承。

### 上下文感知掩膜解码

掩膜解码器在稀疏图上运行，图中的边连接空间邻居和层级相关区域。图注意力传播局部连续性和提示证据，随后二元预测头为每个细粒度区域预测目标概率 $p_k=\sigma(o_k)$。软区域分配随后恢复稠密掩膜：

$$
\hat{M}(x)=\sum_{k=1}^{K_f}A_f(x,k)p_k.
\tag{12}
$$

这种直接概率预测避免了排序截断、逐类别面积先验以及人工选择的分数到掩膜转换规则。推理时我们统一采用 0.5 的阈值。如果正提示和负提示落入同一个硬区域，模型将选择拒答并请求更细化的交互，而不是静默返回含糊的掩膜。

### 分阶段情景式优化

训练遵循模型的依赖结构。我们首先学习稳定的多尺度区域，然后优化细胞感知区域表征和层级交互，之后再引入提示条件情景。每个情景采样一个 WSI 图像块、一种目标组织、一个或多个正提示、可选的困难负提示以及对应的二元掩膜。跨情景改变目标，可以使模型从视觉证据推断任务，而不是依赖固定输出通道。

联合目标函数为：

$$
\begin{aligned}
\mathcal{L}={}&
\mathcal{L}_{\mathrm{BCE}}+\lambda_d\mathcal{L}_{\mathrm{Dice}}
+\lambda_b\mathcal{L}_{\mathrm{bd}}
+\lambda_r\mathcal{L}_{\mathrm{reg}} \\
&+\lambda_a\mathcal{L}_{\mathrm{assign}}
+\lambda_t\mathcal{L}_{\mathrm{task}}.
\end{aligned}
\tag{13}
$$

其中，$\mathcal{L}_{\mathrm{reg}}$ 监督在线区域目标，$\mathcal{L}_{\mathrm{assign}}$ 汇集平衡、熵和紧致度正则项，$\mathcal{L}_{\mathrm{task}}$ 在联合适配期间保持提示表征。像素级 BCE 和 Dice 项优化最终交互目标，边界项则促进精确的组织轮廓。联合微调对预训练图像编码器采用保守更新，并保留区域正则项以避免分配坍缩。

## 参考文献

1. Chan, L.; Hosseini, M. S.; Rowsell, C.; Plataniotis, K. N.; and Damaskinos, S. 2019. HistoSegNet: Semantic Segmentation of Histological Tissue Type in Whole Slide Images. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 10662–10671.

2. Chen, R. J.; Chen, C.; Li, Y.; Chen, T. Y.; Trister, A. D.; Krishnan, R. G.; and Mahmood, F. 2022. Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 16144–16155.

3. Chen, X.; Zhao, Z.; Zhang, Y.; Duan, M.; Qi, D.; and Zhao, H. 2022. FocalClick: Towards Practical Interactive Image Segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 1300–1309.

4. Graham, S.; Vu, Q. D.; Raza, S. E. A.; Azam, A.; Tsang, Y. W.; Kwak, J. T.; and Rajpoot, N. 2019. HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images. *Medical Image Analysis*, 58: 101563.

5. Hörst, F.; Rempe, M.; Heine, L.; Seibold, C.; Keyl, J.; Baldini, G.; Ugurel, S.; Siveke, J.; Grünwald, B.; Egger, J.; and Kleesiek, J. 2024. CellViT: Vision Transformers for Precise Cell Segmentation and Classification. *Medical Image Analysis*, 94: 103143.

6. Jampani, V.; Sun, D.; Liu, M.-Y.; Yang, M.-H.; and Kautz, J. 2018. Superpixel Sampling Networks. In *Proceedings of the European Conference on Computer Vision*, 352–368.

7. Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.-Y.; Dollár, P.; and Girshick, R. 2023. Segment Anything. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 4015–4026.

8. Liu, H.; Yang, H.; van Diest, P. J.; Pluim, J. P. W.; and Veta, M. 2024. WSI-SAM: Multi-Resolution Segment Anything Model for Histopathology Whole-Slide Images. In *Proceedings of the MICCAI Workshop on Computational Pathology*, volume 254 of *Proceedings of Machine Learning Research*, 25–37.

9. Ma, J.; He, Y.; Li, F.; Han, L.; You, C.; and Wang, B. 2024. Segment Anything in Medical Images. *Nature Communications*, 15(1): 654.

10. Maninis, K.-K.; Caelles, S.; Pont-Tuset, J.; and Van Gool, L. 2018. Deep Extreme Cut: From Extreme Points to Object Segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 616–625.

11. Pati, P.; Jaume, G.; Foncubierta-Rodríguez, A.; Feroce, F.; Anniciello, A. M.; Scognamiglio, G.; Brancati, N.; Fiche, M.; Dubruc, E.; Riccio, D.; Di Bonito, M.; De Pietro, G.; Botti, G.; Thiran, J.-P.; Frucci, M.; Goksel, O.; and Gabrani, M. 2022. Hierarchical Graph Representations in Digital Pathology. *Medical Image Analysis*, 75: 102264.

12. Xu, H.; Usuyama, N.; Bagga, J.; Zhang, S.; Rao, R.; Naumann, T.; Wong, C.; Gero, Z.; González, J.; Gu, Y.; Xu, Y.; Wei, M.; Wang, W.; Ma, S.; Wei, F.; Yang, J.; Li, C.; Gao, J.; Rosemon, J.; Bower, T.; Lee, S.; Weerasinghe, R.; Wright, B. J.; Robicsek, A.; Piening, B.; Bifulco, C.; Wang, S.; and Poon, H. 2024. A Whole-Slide Foundation Model for Digital Pathology from Real-World Data. *Nature*, 630: 181–188.
