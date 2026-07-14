# Phase A 多尺度可视化检查报告

- generated_at: 2026-07-09T21:05:17
- visual_root: `/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/visualizations/phase_a_multiscale_check_lowmag_slic`
- index_csv: `/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/visualizations/phase_a_multiscale_check_lowmag_slic/visual_index.csv`
- rendered_images: 20

## 使用方法

- 打开每个 `multiscale_superpixel_gt_check.png`。
- 第一行是 HE + superpixel boundary，第二行是 GT mask + superpixel boundary。
- GT 行里的黑色区域表示该尺度 superpixel 没有覆盖到的像素，即 `superpixels == -1`。
- 横向依次为 small、medium、large，应该看到边界从细到粗变化。
- 如果 large 大量跨越明显不同 GT 类别，说明当前 hierarchical merge 需要改进。
- 如果某类组织内部有大量黑点/黑块，说明 tissue mask 或 superpixel 生成覆盖不足。
