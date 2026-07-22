# AAAI-27 LaTeX Compile Notes

## Compile in the conda environment

From the paper directory:

```bash
cd /home/zhaoyh/CellAtlas/论文/AAAI27_CellAtlas_Paper
conda activate aligner
latexmk -pdf main.tex
```

Equivalent one-line command without activating the environment:

```bash
conda run -n aligner latexmk -pdf main.tex
```

## Clean intermediate files

```bash
conda run -n aligner latexmk -C main.tex
```

## Important AAAI-27 rule

`aaai2027.sty` requires PDFLaTeX. Do not compile this template with XeLaTeX.

Use:

```bash
latexmk -pdf main.tex
```

Do not use:

```bash
latexmk -xelatex main.tex
```

## Add references later

Add BibTeX entries to `references.bib`, cite them in `main.tex`, then uncomment:

```latex
\bibliography{references}
```

near the end of `main.tex`.
