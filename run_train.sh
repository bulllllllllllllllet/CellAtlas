#!/bin/bash

# =================================================================
# 优化后的训练启动脚本
# =================================================================

# 多数据集路径（空格分隔）
CACHE_DIRS=(
    /nfs-medical3/zyh/XCellAligner_feature_extract_output/CRC02
    /nfs-medical3/zyh/XCellAligner_feature_extract_output/CRC03
    /nfs-medical3/zyh/XCellAligner_feature_extract_output/CRC04
    /nfs-medical3/zyh/XCellAligner_feature_extract_output/CRC05
)
START_INDICES=(0 0 0 0)
MIF_CHANNELS=(19 19 19 19)

python multidata_aligner_trainer.py \
    --cache_dir "${CACHE_DIRS[@]}" \
    --start_index "${START_INDICES[@]}" \
    --mif_channel "${MIF_CHANNELS[@]}" \
    --output_dir ./experiments \
    --batch_size 32 \
    --lr 1e-5 \
    --epochs 300 \
    --lambda_mse 100.0 \
    --lambda_contrast 0.1 \
    --max_contrast_cells 512 \
    --max_neg_cells 2048 \
    --num_workers 4

# 参数说明：
# --cache_dir: 多个数据集的特征缓存目录
# --start_index: 每个数据集的起始 patch 索引
# --mif_channel: 每个数据集的 mIF 通道数
# --lambda_mse 100.0: 大幅提高 MSE 权重，强制模型优先学准颜色。
# --lambda_contrast 0.1: 降低对比损失权重，作为辅助。
# --max_contrast_cells 512: 采样 Anchor，防止梯度被海量细胞淹没。
# --max_neg_cells 2048: 采样负样本，降低对比任务难度，提高计算效率。
