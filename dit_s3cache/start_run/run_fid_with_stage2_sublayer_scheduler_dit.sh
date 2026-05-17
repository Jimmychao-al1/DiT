#!/usr/bin/env bash
# DiT sub-layer Stage2 → FID@5K（ADM）
#
# Usage, from DiT repo root:
#   ./dit_s3cache/start_run/run_fid_with_stage2_sublayer_scheduler_dit.sh
#     → default: sweep_K20_sw3_lam1 sub-layer schedule
#   ./dit_s3cache/start_run/run_fid_with_stage2_sublayer_scheduler_dit.sh all
#     → queue K20/K15/K10 sub-layer schedules
#   ./dit_s3cache/start_run/run_fid_with_stage2_sublayer_scheduler_dit.sh --job NAME JSON ...
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

CONDA_ENV="${CONDA_ENV:-ldm}"
if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PY:-python}"
SAMPLE="${SCRIPT_DIR}/sample_stage2_cache_scheduler_sublayer_dit.py"
RESULTS_ROOT="${RESULTS_ROOT:-dit_s3cache/results/fid_dit_stage2_sublayer}"
FID_EXTRA=(--num-fid-samples "${FID_NUM_SAMPLES:-5000}")

BASE_FLAG=()
if [[ "${RUN_STAGE2_CACHE_BASE:-0}" == "1" ]]; then
  BASE_FLAG=(--base)
fi

if [[ "${1:-}" == "all" ]]; then
  shift
  exec "${PY}" "${SAMPLE}" "${BASE_FLAG[@]}" \
    --results-root "${RESULTS_ROOT}" "${FID_EXTRA[@]}" \
    --job "sweep_K20_sw3_lam1_sublayer" "dit_s3cache/stage2/stage2_output_sublayer/src_sweep_K20_sw3_lam1/cache_schedule_sublayer.json" \
    --job "sweep_K15_sw3_lam1_sublayer" "dit_s3cache/stage2/stage2_output_sublayer/src_sweep_K15_sw3_lam1/cache_schedule_sublayer.json" \
    --job "sweep_K10_sw3_lam1_sublayer" "dit_s3cache/stage2/stage2_output_sublayer/src_sweep_K10_sw3_lam1/cache_schedule_sublayer.json" \
    "$@"
fi

if [[ $# -eq 0 ]]; then
  exec "${PY}" "${SAMPLE}" "${BASE_FLAG[@]}" --results-root "${RESULTS_ROOT}" "${FID_EXTRA[@]}"
else
  exec "${PY}" "${SAMPLE}" "${BASE_FLAG[@]}" "$@"
fi
