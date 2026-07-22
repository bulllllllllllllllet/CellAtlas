#!/usr/bin/env bash
# Phase 3 XCell feature worker: persistent log plus explicit pipeline exit status.
set -o pipefail

log_path=$1
shift
conda run -n aligner --no-capture-output python -m benchmarks.v4.phase_3_cell_region.tools.extract_xcell_features "$@" 2>&1 | tee "$log_path"
worker_status=${PIPESTATUS[0]}
echo "EXIT_CODE=${worker_status}" | tee -a "$log_path"
exit "$worker_status"
