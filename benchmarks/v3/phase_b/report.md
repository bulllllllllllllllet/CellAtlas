# Phase B Prompt-task Regeneration

## Summary

Phase B was regenerated from the current `lowmag_loose + slic` multiscale superpixels.
The formal prompt task CSVs do not include the old prompt CSV; it is retained only as `legacy_prompt_audit.csv` with `usage_status=not_used`.

## Outputs

- `auto_prompt_tasks.csv`: /nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/prompt_tasks/auto_prompt_tasks.csv
- `all_prompt_tasks.csv`: /nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/prompt_tasks/all_prompt_tasks.csv
- `prompt_task_summary.csv`: /nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/prompt_tasks/prompt_task_summary.csv
- `phase_b_validation.json`: /nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/reports/phase_b_validation.json
- sample visualizations: /nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner/pret_superpixel/visualizations/phase_b_prompt_task_samples

## Validation

- passed: True
- total_queries: 3711
- unique_query_ids: 3711
- images: 20
- classes: 12
- legacy prompt rows audited, not used: 1161

## Class Coverage

| class | clean | noisy | hard-negative | total |
|---:|---:|---:|---:|---:|
| 0 | 164 | 100 | 190 | 454 |
| 1 | 190 | 100 | 200 | 490 |
| 2 | 200 | 100 | 200 | 500 |
| 3 | 86 | 68 | 110 | 264 |
| 4 | 51 | 81 | 70 | 202 |
| 5 | 0 | 68 | 0 | 68 |
| 6 | 198 | 100 | 200 | 498 |
| 7 | 190 | 100 | 190 | 480 |
| 8 | 68 | 81 | 140 | 289 |
| 9 | 24 | 44 | 50 | 118 |
| 10 | 120 | 67 | 130 | 317 |
| 11 | 2 | 9 | 20 | 31 |
