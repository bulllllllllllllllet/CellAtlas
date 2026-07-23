#!/usr/bin/env bash
set -u

output_dir="/nfs-medical3/zyh/models/wsi_sam_20260722_212128"
stamp="20260722_212128"
max_attempts=20

mkdir -p "$output_dir"

download_with_resume() {
  local file_id="$1"
  local output_name="$2"
  local output_path="$output_dir/$output_name"
  local attempt

  if [[ -s "$output_path" ]]; then
    echo "SKIP existing completed target: $output_path"
    sha256sum "$output_path"
    return 0
  fi

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    echo "DOWNLOAD attempt=$attempt/$max_attempts target=$output_path"
    if gdown --continue "$file_id" -O "$output_path"; then
      echo "COMPLETE target=$output_path bytes=$(stat -c %s "$output_path")"
      sha256sum "$output_path" | tee "$output_path.sha256"
      return 0
    fi
    echo "RETRY target=$output_path after_seconds=30"
    sleep 30
  done

  echo "FAILED target=$output_path attempts=$max_attempts" >&2
  return 1
}

download_with_resume \
  "1rip5UKevAFGoi4QzSltCHjJ7pp4vLtcI" \
  "mobile_sam_${stamp}.pt"
download_with_resume \
  "1ECufMGJfzEHSDbG12ozeulCr0bDw8XX8" \
  "net_high_${stamp}.pth"
download_with_resume \
  "16yfjsdmskv1mArXbENxqgBNUB4LahHQO" \
  "net_low_${stamp}.pth"
download_with_resume \
  "1sKYF82pVo1GsthcTWR0PFBsnTkGES_Bl" \
  "vit_tiny_maskdecoder_${stamp}.pt"

echo "ALL_WSI_SAM_WEIGHTS_COMPLETE output_dir=$output_dir"
