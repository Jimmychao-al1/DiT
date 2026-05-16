#!/usr/bin/env bash
# DiT Stage2 → FID@5K（ADM）
#
# 用法（於 DiT repo 根目錄）：
#   ./dit_s3cache/start_run/run_fid_with_stage2_scheduler_dit.sh overnight
#     → 夜間佇列：跳過已完成的 baseline & P2(kmax4)，其餘 P1/P3 kmax4 + P1/P2/P3 kmax3
#
#   ./dit_s3cache/start_run/run_fid_with_stage2_scheduler_dit.sh
#     → 預設單跑 P2 kmax4 blockwise（與 sample 腳本 default_jobs 一致）
#
#   ./dit_s3cache/start_run/run_fid_with_stage2_scheduler_dit.sh --job NAME JSON ...
#     → 自訂；可加 --num-fid-samples 50000
#
# 環境變數：
#   CONDA_ENV=ldm          （預設）
#   RUN_STAGE2_CACHE_BASE=1  → 加 --base（夜間模式預設不加，因 baseline 已跑過）
#   FID_NUM_SAMPLES=5000   → 傳給 --num-fid-samples（僅 overnight / 無參數預設路徑）
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
SAMPLE="${SCRIPT_DIR}/sample_stage2_cache_scheduler_dit.py"
RESULTS_ROOT="${RESULTS_ROOT:-dit_s3cache/results/fid_dit_stage2}"
FID_EXTRA=(--num-fid-samples "${FID_NUM_SAMPLES:-5000}")

run_sample() {
  "${PY}" "${SAMPLE}" "$@"
}

# ---------------------------------------------------------------------------
# overnight：已完成 baseline + P2(kmax4)；其餘依序排隊
# ---------------------------------------------------------------------------
run_overnight() {
  local log_dir="${ROOT}/${RESULTS_ROOT}"
  mkdir -p "${log_dir}"
  local log_file="${log_dir}/overnight_fid_$(date +%Y%m%d_%H%M%S).log"

  echo "================================================================"
  echo "DiT start_run OVERNIGHT | conda=${CONDA_ENV} | log=${log_file}"
  echo "Started: $(date -Is)"
  echo "================================================================"

  # shellcheck disable=SC2069
  exec > >(tee -a "${log_file}") 2>&1

  local -a jobs=(
    "baseline_p1_K20_sw5_lam1.0_kmax4_blockwise|dit_s3cache/stage2/stage2_output/src_baseline_p1_K20_sw5_lam1.0/02_refined_blockwise/stage2_refined_scheduler_config.json"
    "baseline_p3_K25_sw5_lam1.0_kmax4_blockwise|dit_s3cache/stage2/stage2_output/src_baseline_p3_K25_sw5_lam1.0/02_refined_blockwise/stage2_refined_scheduler_config.json"
    "baseline_p1_K20_sw5_lam1.0_kmax3_blockwise|dit_s3cache/stage2/stage2_output/src_baseline_p1_K20_sw5_lam1.0_kmax3/02_refined_blockwise/stage2_refined_scheduler_config.json"
    "baseline_p2_K20_sw3_lam1.0_kmax3_blockwise|dit_s3cache/stage2/stage2_output/src_baseline_p2_K20_sw3_lam1.0_kmax3/02_refined_blockwise/stage2_refined_scheduler_config.json"
    "baseline_p3_K25_sw5_lam1.0_kmax3_blockwise|dit_s3cache/stage2/stage2_output/src_baseline_p3_K25_sw5_lam1.0_kmax3/02_refined_blockwise/stage2_refined_scheduler_config.json"
  )

  local -a py_args=(--results-root "${RESULTS_ROOT}" "${FID_EXTRA[@]}")
  for entry in "${jobs[@]}"; do
    IFS='|' read -r name json <<< "${entry}"
    if [[ ! -f "${ROOT}/${json}" ]]; then
      echo "[ERROR] Missing scheduler JSON: ${json}" >&2
      exit 1
    fi
    py_args+=(--job "${name}" "${json}")
  done

  echo "Jobs queued: ${#jobs[@]} (skip: dit_baseline_full_compute, baseline_p2 kmax4)"
  printf '  - %s\n' "${jobs[@]%%|*}"

  run_sample "${py_args[@]}"

  echo "================================================================"
  echo "OVERNIGHT complete: $(date -Is)"
  echo "Log: ${log_file}"
  echo "Index: ${RESULTS_ROOT}/runs_index.jsonl"
  echo "================================================================"
}

# ---------------------------------------------------------------------------
# 一般模式
# ---------------------------------------------------------------------------
BASE_FLAG=()
if [[ "${RUN_STAGE2_CACHE_BASE:-0}" == "1" ]]; then
  BASE_FLAG=(--base)
fi

if [[ "${1:-}" == "overnight" ]]; then
  shift
  run_overnight "$@"
  exit 0
fi

if [[ $# -eq 0 ]]; then
  exec "${PY}" "${SAMPLE}" "${BASE_FLAG[@]}" "${FID_EXTRA[@]}"
else
  exec "${PY}" "${SAMPLE}" "${BASE_FLAG[@]}" "$@"
fi
