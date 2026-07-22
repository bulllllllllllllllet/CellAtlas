#!/usr/bin/env bash
# Persistent Phase 3 DDP training launcher: one log, explicit final exit code.
set -o pipefail
log_path=$1
shift
CUDA_VISIBLE_DEVICES=1,3,4,5 conda run -n aligner --no-capture-output torchrun --standalone --nproc_per_node=4 -m benchmarks.v4.phase_3_cell_region.train_phase3 "$@" 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
echo "EXIT_CODE=${status}" | tee -a "$log_path"
exit "$status"
