#!/usr/bin/env bash
# Stage-2：P1 / P2 / P3 的 kmax3 Stage1 排程 → global + blockwise refine
#
# 用法（DiT repo 根目錄，conda 預設 ldm）：
#   bash dit_s3cache/stage2/run_stage2_kmax3_all_dit.sh
# 只跑單一 source：
#   bash dit_s3cache/stage2/run_stage2_full_experiments_dit.sh baseline_p1_K20_sw5_lam1.0_kmax3
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RUNNER="dit_s3cache/stage2/run_stage2_full_experiments_dit.sh"

SOURCES=(
  "baseline_p1_K20_sw5_lam1.0_kmax3"
  "baseline_p2_K20_sw3_lam1.0_kmax3"
  "baseline_p3_K25_sw5_lam1.0_kmax3"
)

if [[ $# -ge 1 ]]; then
  bash "${RUNNER}" "$@"
  exit 0
fi

for src in "${SOURCES[@]}"; do
  bash "${RUNNER}" "${src}"
done

echo "✅ Stage-2 kmax3（P1/P2/P3）全部完成 → dit_s3cache/stage2/stage2_output/src_baseline_p*_kmax3/"
