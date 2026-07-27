#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

pdflatex -interaction=nonstopmode -halt-on-error \
  detached_experiment_figures.tex

names=(
  table_patch_comparison
  figure_prompt_budget_curve
  table_wsi_comparison
  table_wsi_per_class
  table_ablation
)

for page in 1 2 3 4 5; do
  read -r llx lly urx ury < <(
    gs -q -dNOPAUSE -dBATCH \
      -dFirstPage="$page" -dLastPage="$page" \
      -sDEVICE=bbox detached_experiment_figures.pdf 2>&1 |
      awk '/HiResBoundingBox/{print $2, $3, $4, $5}'
  )

  read -r crop_x crop_y crop_w crop_h < <(
    awk -v x1="$llx" -v y1="$lly" -v x2="$urx" -v y2="$ury" \
      'BEGIN {
         margin = 4
         printf "%.3f %.3f %.3f %.3f\n",
                x1 - margin, y1 - margin,
                x2 - x1 + 2 * margin, y2 - y1 + 2 * margin
       }'
  )

  index=$((page - 1))
  gs -q -dNOPAUSE -dBATCH \
    -dFirstPage="$page" -dLastPage="$page" \
    -sDEVICE=pdfwrite \
    -dCompatibilityLevel=1.7 \
    -dFIXEDMEDIA \
    -dDEVICEWIDTHPOINTS="$crop_w" \
    -dDEVICEHEIGHTPOINTS="$crop_h" \
    -sOutputFile="${names[$index]}.pdf" \
    -c "<</PageOffset [-$crop_x -$crop_y]>> setpagedevice" \
    -f detached_experiment_figures.pdf
done

printf '%s\n' \
  "Updated detached_experiment_figures.pdf" \
  "Updated table_patch_comparison.pdf" \
  "Updated figure_prompt_budget_curve.pdf" \
  "Updated table_wsi_comparison.pdf" \
  "Updated table_wsi_per_class.pdf" \
  "Updated table_ablation.pdf"
