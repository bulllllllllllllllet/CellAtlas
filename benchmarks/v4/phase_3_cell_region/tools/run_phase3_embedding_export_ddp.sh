#!/usr/bin/env bash
# Persistent Phase 3 embedding exporter: explicit GPU set, one log and exit code.
set -o pipefail
log_path=$1
gpu_ids=$2
process_count=$3
shift 3
CUDA_VISIBLE_DEVICES="$gpu_ids" conda run -n aligner --no-capture-output \
  torchrun --standalone --nproc_per_node="$process_count" \
  -m benchmarks.v4.phase_3_cell_region.tools.export_phase3_embeddings "$@" 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
echo "EXIT_CODE=${status}" | tee -a "$log_path"
exit "$status"
