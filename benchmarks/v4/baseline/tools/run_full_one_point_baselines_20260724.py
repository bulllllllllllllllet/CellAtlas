import subprocess
from pathlib import Path

root=Path('/home/zhaoyh/CellAtlas'); manifest='/nfs-medical3/zyh/v4/baseline/one_point_one_negative_manifest_20260724_000100/episode_manifest_20260724_000100.parquet'
for config,stamp in [('sam_zero_shot.yaml','20260724_000200'),('sam_med2d_zero_shot.yaml','20260724_000201'),('wsi_sam_zero_shot.yaml','20260724_000202')]:
 subprocess.run(['conda','run','-n','aligner','python','-m','benchmarks.v4.baseline.tools.evaluate_baseline','--config',f'benchmarks/v4/baseline/configs/{config}','--phase2-config','benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml','--episode-manifest',manifest,'--split','val','--timestamp',stamp,'--gpus','0','--num-workers','0','--batch-size','1','--shard-size','32'],cwd=root,check=True)
