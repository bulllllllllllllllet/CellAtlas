# Phase A 多尺度 superpixel / region token 报告

- generated_at: 2026-07-09T18:08:13
- output_root: `/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner`
- data_manifest: `/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/data_manifest_v3.csv`
- generation_mode: slic
- total_image_scale_outputs: 60
- passed: true

## 尺度统计

### small

- image_count: 20
- segment_count_min/median/max: 6259/8333.5/12606
- adjacency_edges_min/median/max: 18225/24457.0/37040
- base_token_shape_example: [7518, 274]
- enhanced_token_shape_example: [7518, 309]

### medium

- image_count: 20
- segment_count_min/median/max: 1565/2089.0/3152
- adjacency_edges_min/median/max: 4432/6011.0/9098
- base_token_shape_example: [1879, 274]
- enhanced_token_shape_example: [1879, 309]

### large

- image_count: 20
- segment_count_min/median/max: 993/999.0/1000
- adjacency_edges_min/median/max: 2747/2851.0/2874
- base_token_shape_example: [1000, 274]
- enhanced_token_shape_example: [1000, 309]

## 输出说明

- `multiscale_tokens/<image_id>/<scale>/tokens_image_cell_reg.npy` 是主方法 token，指向 `image_cell_reg_cellw0p5`。
- `adjacency.npy` 保存无向邻接边列表，shape 为 `[edge_count, 2]`，不是稠密邻接矩阵。
- `small` / `medium` / `large` 均使用 `generation_mode=slic` 全量重新生成。
- tissue mask 使用 `tissue_mask_mode=lowmag_loose`：低倍率生成宽松组织 mask，再限制各尺度 SLIC 只在 mask 内运行。
- 每个尺度同时保存 `he_tissue_mask.npy`、`he_tissue_high_conf.npy`、`he_tissue_low_conf.npy`，其中 low-conf mask 作为后续高倍复核候选区域。
