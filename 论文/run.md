  AAAI-27 模板强制要求 PDFLaTeX，不能用 XeLaTeX。以后你这样编译：

  cd /home/zhaoyh/CellAtlas/论文/AAAI27_CellAtlas_Paper
  conda activate aligner
  latexmk -pdf main.tex

  或者不激活 conda，直接：

  cd /home/zhaoyh/CellAtlas/论文/AAAI27_CellAtlas_Paper
  conda run -n aligner latexmk -pdf main.tex

  清理中间文件：

  conda run -n aligner latexmk -C main.tex