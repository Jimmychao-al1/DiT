#!/usr/bin/env bash
# Stage-1 baseline sweep（DiT 版）：K（change points 個數）、smooth_window、lambda、k_max
#
# 用法：
#   bash dit_s3cache/stage1/run_stage1_sweep_dit.sh
# 或覆寫路徑與陣列：
#   STAGE0_DIR=... BASE_OUT=... bash dit_s3cache/stage1/run_stage1_sweep_dit.sh
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE0_DIR="${STAGE0_DIR:-dit_s3cache/stage0/stage0_output}"
BASE_OUT="${BASE_OUT:-dit_s3cache/stage1/stage1_output}"
BASE_FIG="${BASE_FIG:-dit_s3cache/stage1/stage1_figures}"
SCHEDULER="dit_s3cache/stage1/stage1_scheduler_dit.py"
VISUALIZE="dit_s3cache/stage1/visualize_stage1_dit.py"
VERIFY="dit_s3cache/stage1/verify_scheduler_dit.py"

# 掃描範圍（可自行縮小以縮短時間）
K_LIST=(10 15 20 25 30 35)
SW_LIST=(2 3 5 9)
LAM_LIST=(0.25 0.5 1.0)
KMAX_LIST=(4)

echo "================================================================"
echo "Stage-1 baseline sweep（DiT-XL/2 256×256，T=250 DDPM）"
echo "STAGE0_DIR=${STAGE0_DIR}"
echo "K_LIST=( ${K_LIST[*]} )"
echo "SW_LIST=( ${SW_LIST[*]} )"
echo "LAM_LIST=( ${LAM_LIST[*]} )"
echo "KMAX_LIST=( ${KMAX_LIST[*]} )"
echo "================================================================"

for K in "${K_LIST[@]}"; do
  for SW in "${SW_LIST[@]}"; do
    for LAM in "${LAM_LIST[@]}"; do
      for KMAX in "${KMAX_LIST[@]}"; do
        TAG="K${K}_sw${SW}_lam${LAM}_kmax${KMAX}"
        OUT_DIR="${BASE_OUT}/sweep_${TAG}"
        FIG_DIR="${BASE_FIG}/sweep_${TAG}"
        mkdir -p "${OUT_DIR}" "${FIG_DIR}"

        echo "────────────────────────────────────────"
        echo "▶ ${TAG}"
        echo "  output : ${OUT_DIR}"
        echo "────────────────────────────────────────"

        python3 "${SCHEDULER}" \
          --stage0_dir "${STAGE0_DIR}" \
          --output_dir "${OUT_DIR}" \
          --K "${K}" \
          --smooth_window "${SW}" \
          --lambda "${LAM}" \
          --k_max "${KMAX}"

        python3 "${VERIFY}" --config "${OUT_DIR}/scheduler_config.json"
        python3 "${VISUALIZE}" --stage1_output_dir "${OUT_DIR}" --output_dir "${FIG_DIR}"
        echo ""
      done
    done
  done
done

echo "================================================================"
echo "▶ 匯出 CSV（與 LDM csv_exports 相同欄位）"
python3 dit_s3cache/stage1/export_stage1_sweep_csv_dit.py \
  --base_out "${BASE_OUT}"

echo "================================================================"
echo "✅ sweep 完成。結果：${BASE_OUT}/sweep_*"
echo "   CSV：${BASE_OUT}/csv_exports/"
echo "================================================================"
