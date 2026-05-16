#!/usr/bin/env bash
# DiT Stage2 scheduler → FID@5K，預設 P2 blockwise。
# 用法：於 DiT repo 根目錄 ./dit_s3cache/start_run/run_fid_with_stage2_scheduler_dit.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

BASE_FLAG=()
if [[ "${RUN_STAGE2_CACHE_BASE:-0}" == "1" ]]; then
  BASE_FLAG=(--base)
fi

EXTRA=("$@")

exec python3 "${SCRIPT_DIR}/sample_stage2_cache_scheduler_dit.py" \
  "${BASE_FLAG[@]}" \
  "${EXTRA[@]}"
