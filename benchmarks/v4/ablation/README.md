# V4 Ablation Code

This directory contains experiment-specific configurations and launch helpers.
Core model implementations remain in their owning Phase-4 and Phase-6 modules;
the helpers here enforce a common J5 initialization, validation-only selection,
and timestamped NFS outputs.

Available P0 experiments:

- `phase4_attention.yaml`: content-adaptive sparse parent attention (A2).
- `phase4_attention_shuffled.yaml`: matched shuffled-parent control (A3).
- `cell_branch_removed.yaml`: Fine visual token without Phase-3 cell fusion (A6).
- `phase2_fixed_slic.yaml`: fixed SLIC assignment replacing the learned Phase-2 assignment (A8).
- `prompt_mean_prototype.yaml`: masked positive/negative mean prototype replacing the Phase-5 Set Encoder (A10).

Use `tools/launch_ablation.py --help` to print or launch a command.  It refuses
to use the sealed test split and validates that every P0 experiment starts from
the J5 validation-best checkpoint.

## Results

All reported model-selection metrics below use the fixed validation-only
4,000-episode manifest.  The reference is J5 epoch 4: Pixel macro/micro Dice
`0.74132793 / 0.81380071`.  No sealed-test split was read.

| ID | Route | Status | Pixel macro Dice | Pixel micro Dice | Boundary F1 | Utilization | Decision / artifacts |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| A2 | Content-adaptive sparse parent attention; physical top-4 parent edges | Complete, `EXIT_CODE=0` | 0.73781489 (-0.00351304) | 0.81335673 (-0.00044398) | 0.31232945 | attention entropy 0.86221; effective parents 4.0; residual/Fine norm 0.66706 | No eligible checkpoint: both Dice values are below J5 and neither improves. Run: `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260725_154040`; log: `/nfs-medical3/zyh/v4/phase6/logs/phase4_attention_physical_formal_20260725_154040.log`. |
| A3 | Same attention model with deterministic shuffled parent identities | Complete, `EXIT_CODE=0` | 0.73752222 (-0.00380571) | 0.81279651 (-0.00100420) | 0.31219047 | attention entropy 0.87641; effective parents 4.0; residual/Fine norm 0.67715 | Real physical edges exceed shuffled edges by only 0.00029267 macro Dice; neither route improves on J5. Run: `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260725_215342`; log: `/nfs-medical3/zyh/v4/phase6/logs/phase4_attention_shuffled_formal_20260725_215342.log`. |
| A6 | Fine visual token without Phase-3 cell fusion; Prompt/Decoder retrained | Complete, `EXIT_CODE=0` | 0.58846192 (-0.15286601) | 0.65430440 (-0.15949631) | 0.24708353 | Fine visual region token only; no cell fusion | Clear removal ablation: removing Phase-3 cell-aware fusion severely degrades all Pixel metrics. Run: `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260726_105600`; log: `/nfs-medical3/zyh/v4/phase6/logs/phase3_cell_removed_formal_20260726_105600.log`. |
| A8 | Fixed 64-slot SLIC assignment replacing learned Phase-2 regionization; cell/Prompt/Decoder retrained | Complete, `EXIT_CODE=0` | 0.56502706 (-0.17630086) | 0.64815249 (-0.16564821) | 0.14754451 | prompt-conflict rate 0.5645; epoch 2 is the strongest macro-Dice checkpoint | Clear removal ablation: fixed SLIC causes a large Pixel and boundary degradation, so learned regionization is necessary. No eligible checkpoint. Run: `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260726_150700`; log: `/nfs-medical3/zyh/v4/phase6/logs/phase2_fixed_slic_formal_20260726_150700.log`. |
| A10 | Masked positive/negative mean prototype replacing the Phase-5 Prompt Set Encoder; prompt tail/Decoder retrained | Complete, `EXIT_CODE=0` | 0.72146738 (-0.01986055) | 0.79700338 (-0.01679733) | 0.30624106 | prompt-conflict rate 0.05; 3 GPUs, global batch 6; epoch 4 | Clear module ablation: replacing learned set aggregation with mean prototypes degrades both Pixel Dice metrics, so the Prompt Set Encoder is necessary. No eligible checkpoint. Run: `/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260727_071422`; log: `/nfs-medical3/zyh/v4/phase6/logs/prompt_mean_prototype_formal_3gpu_20260727_071422.log`. |
