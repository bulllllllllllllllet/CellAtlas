import json
import subprocess

metadata = json.load(open('/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260723_225436/metadata.json'))
command = metadata['summary']['reproducibility']['command']
command[1:2] = ['-m', 'benchmarks.v4.phase_6_mask_decoder.tools.evaluate_visualize_joint_pixel']
command[command.index('--timestamp') + 1] = '20260723_232000'
subprocess.run(command, check=True, cwd="/home/zhaoyh/CellAtlas")
