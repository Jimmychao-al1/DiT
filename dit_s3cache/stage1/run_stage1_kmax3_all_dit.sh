#!/usr/bin/env bash
# Stage-1：P1 / P2 / P3，k_max=3（其餘參數與各 priority 的 kmax4 主線一致）
#
# 用法（DiT repo 根目錄）：
#   bash dit_s3cache/stage1/run_stage1_kmax3_all_dit.sh
# 只跑單一 priority（1|2|3）：
#   ONLY_P=2 bash dit_s3cache/stage1/run_stage1_kmax3_all_dit.sh
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE0_DIR="${STAGE0_DIR:-dit_s3cache/stage0/stage0_output}"
BASE_OUT="${BASE_OUT:-dit_s3cache/stage1/stage1_output}"
BASE_FIG="${BASE_FIG:-dit_s3cache/stage1/stage1_figures}"
SCHEDULER="dit_s3cache/stage1/stage1_scheduler_dit.py"
VISUALIZE="dit_s3cache/stage1/visualize_stage1_dit.py"
VERIFY="dit_s3cache/stage1/verify_scheduler_dit.py"

run_one() {
  local run_name="$1"
  local k="$2"
  local sw="$3"
  local lam="$4"
  local kmax="$5"

  local out_dir="${BASE_OUT}/${run_name}"
  local fig_dir="${BASE_FIG}/${run_name}"

  echo "================================================================"
  echo "Stage-1: ${run_name}"
  echo "  K=${k} smooth_window=${sw} lambda=${lam} k_max=${kmax}"
  echo "  output : ${out_dir}"
  echo "================================================================"

  mkdir -p "${out_dir}" "${fig_dir}"

  python3 "${SCHEDULER}" \
    --stage0_dir "${STAGE0_DIR}" \
    --output_dir "${out_dir}" \
    --K "${k}" \
    --smooth_window "${sw}" \
    --lambda "${lam}" \
    --k_max "${kmax}"

  python3 "${VERIFY}" --config "${out_dir}/scheduler_config.json"
  python3 "${VISUALIZE}" --stage1_output_dir "${out_dir}" --output_dir "${fig_dir}"
  echo ""
}

# P1 / P2 / P3（對齊 kmax4 的 K、sw、lam）
declare -a JOBS=(
  "baseline_p1_K20_sw5_lam1.0_kmax3|20|5|1.0|3"
  "baseline_p2_K20_sw3_lam1.0_kmax3|20|3|1.0|3"
  "baseline_p3_K25_sw5_lam1.0_kmax3|25|5|1.0|3"
)

for entry in "${JOBS[@]}"; do
  IFS='|' read -r name k sw lam kmax <<< "${entry}"
  prio="${name#baseline_p}"
  prio="${prio%%_*}"
  if [[ -n "${ONLY_P:-}" && "${prio}" != "${ONLY_P}" ]]; then
    continue
  fi
  run_one "${name}" "${k}" "${sw}" "${lam}" "${kmax}"
done

echo "================================================================"
echo "✅ Stage-1 kmax3（P1/P2/P3）完成 → ${BASE_OUT}/baseline_p*_kmax3"
echo "================================================================"
