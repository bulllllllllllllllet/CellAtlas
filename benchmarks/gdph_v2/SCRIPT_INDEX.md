# GDPH v2 Script Index

这个目录里的脚本很多入口被 `python -m benchmarks.gdph_v2.<module>` 和 shell runner 直接引用。为避免破坏现有命令，这里只做逻辑归类，不物理移动文件。

## Core Benchmark Pipeline

| Script | Purpose |
|---|---|
| `init_experiment.py` | 初始化 GDPH benchmark 输出目录和基础配置 |
| `fullres_inference.py` | WSI full-resolution cell inference / feature cache |
| `run_remaining_pipeline.py` | fullres 后续 pipeline 汇总入口 |
| `generate_queries.py` | 生成 query-by-region retrieval queries |
| `eval_retrieval.py` | 细胞/区域 retrieval 指标 |
| `eval_patch_retrieval.py` | patch/hybrid retrieval 指标 |
| `eval_crossval.py` | cross-validation / linear probe 相关评估 |
| `eval_cell_coverage.py` | cell coverage 统计 |
| `eval_nucleus_detection.py` | nucleus/cell detection 评估 |
| `polygon_labels.py` | polygon label 处理 |
| `generate_final_report.py` | 原始 GDPH benchmark 最终报告 |
| `audit_experiment.py` | 实验输出审计 |

## PRET Superpixel Pipeline

| Script | Purpose |
|---|---|
| `pret_superpixel_tokens.py` | HE-derived superpixel token 生成 |
| `pret_generate_prompts.py` | realistic/oracle/scribble prompt 生成 |
| `pret_eval_in_context.py` | PRET-style prototype retrieval baseline |
| `pret_visualize.py` | 原始 PRET 可视化 |
| `pret_build_canonical_prompts.py` | canonical prompt 构建 |
| `pret_compare_scales.py` | 4096/8192/full10x scale 对齐分析 |
| `pret_analyze_results.py` | PRET 结果分析 |
| `pret_repair_eval_outputs.py` | 修复/补全 PRET eval 输出 |
| `pret_make_evidence_pack.py` | 汇总 evidence pack |
| `run_pret_superpixel_scale.sh` | scale experiment runner |

## AAAI PRET Experiments

| Script | Purpose |
|---|---|
| `pret_aaai_eval.py` | AAAI baseline eval：threshold/prototype/patch/image/cell |
| `pret_aaai_sam_baseline.py` | SAM / MedSAM box-prompt baseline |
| `pret_aaai_report.py` | AAAI baseline report |
| `pret_aaai_enhance_tokens.py` | 在已有 superpixel 上补 H/E texture + cell-stat 增强 token |
| `pret_aaai_next_experiments.py` | area calibration + multi-positive/negative interaction curve |
| `pret_aaai_supervised_upper_bound.py` | supervised LOIO token upper bound |
| `pret_aaai_next_report.py` | next interaction 简表报告 |
| `pret_aaai_final_outputs.py` | final main table、delta table、LOIO audit、AAAI Results |
| `pret_aaai_visual_summary.py` | 每类 best/median/worst/hard-negative 可视化；默认只保留 `00_combined_grid.png`、legend、summary |
| `pret_aaai_combine_visual_grids.py` | 把已有 visual-summary case 的 8 张 PNG 拼成 `00_combined_grid.png`；用于兼容旧输出 |
| `run_pret_aaai_baselines.sh` | AAAI baseline runner |
| `run_pret_aaai_next.sh` | next interaction runner |
| `run_pret_aaai_final_outputs.sh` | 最终 AAAI 输出一键 runner |

## Utilities / Setup / Validation

| Script | Purpose |
|---|---|
| `experiment.py` | benchmark path/config constants |
| `geometry.py` | polygon/geometry utility |
| `patch_geometry.py` | patch geometry utility |
| `patch_inference.py` | patch-level inference helper |
| `run_logged.py` | command logging helper |
| `gpu_preflight.py` | generic GPU check |
| `pret_gpu_preflight.py` | PRET/AAAI GPU check |
| `validate_setup.py` | setup validation |
| `validate_model_compatibility.py` | model/checkpoint compatibility validation |
| `wait_for_stage.py` | stage wait utility |
| `select_main.py` | main split selection |
| `select_pilot.py` | pilot split selection |
| `smoke_test_fullres.py` | fullres smoke test |

## Tests

测试位于 `benchmarks/gdph_v2/tests/`。当前主要是脚本级单元测试和 smoke 逻辑测试，没有统一 CI。

## Recommended Entrypoints

| Task | Command |
|---|---|
| Full AAAI final outputs v2 | `NEXT_ROOT=/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v2_full10x benchmarks/gdph_v2/run_pret_aaai_final_outputs.sh` |
| Full AAAI final outputs with enhanced token | `PRIMARY_VARIANT=image_cell_reg_texture_cellstats NEXT_ROOT=/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v2_full10x_texture benchmarks/gdph_v2/run_pret_aaai_final_outputs.sh` |
| Rebuild combined visual grids only | `python -m benchmarks.gdph_v2.pret_aaai_combine_visual_grids --next_root /nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_next_full10x` |
| Generate final result tables only | `python -m benchmarks.gdph_v2.pret_aaai_final_outputs --next_root /nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v2_full10x` |
| Check final report | `/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v2_full10x/pret_superpixel/AAAI_RESULTS.md` |
| Check visual examples | `/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v2_full10x/pret_superpixel/visual_summary/` |
