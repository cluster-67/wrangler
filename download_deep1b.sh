#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
  echo "SCRATCH is not set. Please set SCRATCH to your scratch directory." >&2
  exit 1
fi

dest_dir="${SCRATCH}/data/deep1b"
mkdir -p "${dest_dir}"

base_url="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP"
files=(
  # "base.10M.fbin"
  "base.1B.fbin"
  # "query.public.10K.fbin"
  # "groundtruth.public.10K.ibin"
)

for filename in "${files[@]}"; do
  aria2c \
    -x 16 \
    -s 16 \
    -k 16M \
    -c \
    -d "${dest_dir}" \
    "${base_url}/${filename}"
done
