# Phase A 覆盖修复报告

- generated_at: 2026-07-08T12:59:18
- output_root: `/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner`
- repaired_image_scales: 3
- added_segments: 6874
- missing_pixels_before: 76687920
- missing_pixels_after: 0

## 修复方法

- 只修复 `GT != background` 且 `superpixels == -1` 的区域。
- 未覆盖区域按当前尺度典型 superpixel 尺寸切成 grid segment。
- 新增 segment 的 token 继承最近已有 segment 的 token。
- 背景区域保持 `-1`，不强行覆盖玻片空白。
