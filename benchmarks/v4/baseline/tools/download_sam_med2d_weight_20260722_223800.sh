#!/usr/bin/env bash
set -u

repo_id="schengal1/SAM-Med2D_model"
filename="sam-med2d_b.pth"
local_dir="/nfs-medical3/zyh/models/sam_med2d_20260722_212128/hf_xet_20260722_212128"
target="$local_dir/$filename"
expected_bytes="2561829701"
expected_sha256="921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18"
max_attempts=20

mkdir -p "$local_dir"

validate_target() {
  local actual_bytes actual_sha256
  [[ -f "$target" ]] || return 1
  actual_bytes="$(stat -c %s "$target")"
  actual_sha256="$(sha256sum "$target" | awk '{print $1}')"
  [[ "$actual_bytes" == "$expected_bytes" ]] || return 1
  [[ "$actual_sha256" == "$expected_sha256" ]] || return 1
  printf '%s  %s\n' "$actual_sha256" "$target" | tee "$target.sha256"
}

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  if validate_target; then
    echo "SAM_MED2D_COMPLETE target=$target bytes=$expected_bytes"
    exit 0
  fi

  echo "DOWNLOAD attempt=$attempt/$max_attempts target=$target"
  HF_XET_HIGH_PERFORMANCE=1 conda run -n aligner python -c \
    "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='$repo_id', filename='$filename', local_dir='$local_dir'))" || true

  if validate_target; then
    echo "SAM_MED2D_COMPLETE target=$target bytes=$expected_bytes"
    exit 0
  fi

  echo "RETRY target=$target after_seconds=30"
  sleep 30
done

echo "SAM_MED2D_FAILED target=$target attempts=$max_attempts" >&2
exit 1
