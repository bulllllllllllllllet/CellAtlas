# FewClick AAAI 2027 英文化交付说明

## 文件

- `FewClick_AAAI27.tex`：按 AAAI 2027 匿名投稿模板排版的英文主稿。
- `FewClick_AAAI27.bib`：主稿使用的 BibTeX 文献库。
- `FewClick_AAAI27.pdf`：已编译的预览稿。
- 原中文 Markdown 未修改。

## 编译

从当前目录运行：

```bash
TEXINPUTS='../AuthorKit27/AuthorKit27:' BSTINPUTS='../AuthorKit27/AuthorKit27:' pdflatex FewClick_AAAI27.tex
BSTINPUTS='../AuthorKit27/AuthorKit27:' bibtex FewClick_AAAI27
TEXINPUTS='../AuthorKit27/AuthorKit27:' BSTINPUTS='../AuthorKit27/AuthorKit27:' pdflatex FewClick_AAAI27.tex
TEXINPUTS='../AuthorKit27/AuthorKit27:' BSTINPUTS='../AuthorKit27/AuthorKit27:' pdflatex FewClick_AAAI27.tex
```

## 尚需作者补充

主稿用 `AUTHOR INPUT NEEDED` 标出了不可凭空补写的信息：

1. 摘要中的数据集、队列规模、提示协议和主要量化结果。
2. 完整实验章节：数据与划分、标注、实现细节、提示模拟、基线、指标、主结果、消融、效率、统计分析和定性结果。
3. TLS 跨数据集实验的数据集名称、样本量、提示设置、精确 Dice 与不确定性。
4. 经实验支持的结论与贡献表述。
5. 作者、单位、邮箱，以及终稿阶段需要替换的 camera-ready 设置。
6. 方法图、定性结果图和实验表格；中文源稿没有提供这些素材。

## 已修复的源稿问题

- 删除了方法 3.5 的整段重复文本。
- 将重复的“3.8 推理流程”调整为独立的 Inference 小节。
- 恢复了缺少左端项的细胞上下文、融合表示、层级表示和掩膜投影公式。
- 修复了 Markdown 转义造成的集合、下标、乘号和公式括号错误。
- 将五条贡献压缩为四条清晰贡献，未加入未经实验验证的新主张。
- 参考文献改为 AAAI `aaai2027.bst` 管理的 BibTeX 引用。

## 当前校验状态

- PDFLaTeX + BibTeX 编译通过。
- PDF 为 US Letter、双栏、6 页、PDF 1.7。
- 所有字体均已嵌入，无 Type 3 字体。
- 无未解析引用、LaTeX 错误或水平溢出。
- 仅有首页约 0.78 pt 的模板级纵向 overfull 提示。
- 已扫描常见 Chinglish、AI 高频套话和中文字符，正文未检出高风险项。

注意：当前 6 页主要是因为中文源稿的实验章节为空。补齐实验、图表后页数会明显增加，需要依据 AAAI 2027 主会最终公布的正文页数限制进一步压缩。


 cd /home/zhaoyh/CellAtlas/论文/AAAI27_FewClick_paper
  latexmk -pdf -interaction=nonstopmode -halt-on-error FewClick_AAAI27.tex

  ./build_detached_experiment_figures.sh