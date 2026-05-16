#!/bin/bash
# DiT S3-Cache Stage 2: Full experiment pipeline
# Usage: bash dit_s3cache/stage2/run_stage2_full_experiments_dit.sh [SOURCE_NAME]
#
# If SOURCE_NAME is provided, only that source will be processed.
# Otherwise all SOURCES listed below will be processed in order.
#
# Example (single source):
#   bash dit_s3cache/stage2/run_stage2_full_experiments_dit.sh baseline_p1_K20_sw5_lam1.0_kmax4
#
# Conda：預設啟用環境 ldm（可覆寫 CONDA_ENV=...）

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CONDA_ENV="${CONDA_ENV:-ldm}"
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
export PYTHONPATH="${PWD}"
PY="${PY:-python}"

STAGE1_ROOT="dit_s3cache/stage1/stage1_output"
STAGE2_ROOT="dit_s3cache/stage2/stage2_output"

# Stage 1 候選組合（使用 baseline_p* 實際目錄名）
SOURCES=(
    "baseline_p1_K20_sw5_lam1.0_kmax4"     # P1 主線
    "baseline_p2_K20_sw3_lam1.0_kmax4"     # P2
    "baseline_p3_K25_sw5_lam1.0_kmax4"     # P3 fallback
)

# 如果有傳入參數，只跑指定的 source
if [ $# -ge 1 ]; then
    SOURCES=("$1")
fi

for SRC in "${SOURCES[@]}"; do
    echo "=========================================="
    echo "Processing: ${SRC}"
    echo "=========================================="

    # Stage2 輸出：完整 Stage1 目錄名（含 kmax），避免 kmax3/kmax4 互相覆蓋
    OUT_DIR="${STAGE2_ROOT}/src_${SRC}"
    STAGE1_JSON="${STAGE1_ROOT}/${SRC}/scheduler_config.json"

    if [ ! -f "${STAGE1_JSON}" ]; then
        echo "[ERROR] Stage1 config not found: ${STAGE1_JSON}"
        exit 1
    fi

    mkdir -p "${OUT_DIR}/00_global_refine"
    mkdir -p "${OUT_DIR}/01_blockwise_threshold"
    mkdir -p "${OUT_DIR}/02_refined_blockwise"

    # ------------------------------------------
    # Pass 1: Global refine
    # ------------------------------------------
    echo ""
    echo "[Pass 1] Global refine → ${OUT_DIR}/00_global_refine"
    "${PY}" dit_s3cache/stage2/stage2_runtime_refine_dit.py \
        --scheduler-config "${STAGE1_JSON}" \
        --threshold-mode global \
        --output-dir "${OUT_DIR}/00_global_refine" \
        --eval-num-images 8 \
        --eval-chunk-size 1 \
        --seed 42

    # ------------------------------------------
    # Build blockwise thresholds
    # ------------------------------------------
    echo ""
    echo "[Threshold] Building blockwise thresholds → ${OUT_DIR}/01_blockwise_threshold"
    "${PY}" dit_s3cache/stage2/build_blockwise_thresholds_dit.py \
        --diagnostics "${OUT_DIR}/00_global_refine/stage2_runtime_diagnostics.json" \
        --output "${OUT_DIR}/01_blockwise_threshold/stage2_thresholds_blockwise.json" \
        --q-zone 0.90 \
        --q-peak 0.80 \
        --peak-over-zone-ratio-min 1.3

    # ------------------------------------------
    # Pass 2: Blockwise refine
    # ------------------------------------------
    echo ""
    echo "[Pass 2] Blockwise refine → ${OUT_DIR}/02_refined_blockwise"
    "${PY}" dit_s3cache/stage2/stage2_runtime_refine_dit.py \
        --scheduler-config "${STAGE1_JSON}" \
        --threshold-mode blockwise \
        --threshold-config "${OUT_DIR}/01_blockwise_threshold/stage2_thresholds_blockwise.json" \
        --output-dir "${OUT_DIR}/02_refined_blockwise" \
        --eval-num-images 8 \
        --eval-chunk-size 1 \
        --seed 42

    # ------------------------------------------
    # Verify
    # ------------------------------------------
    echo ""
    echo "[Verify] Checking refined scheduler config..."
    "${PY}" dit_s3cache/stage2/verify_stage2_dit.py \
        "${OUT_DIR}/02_refined_blockwise/stage2_refined_scheduler_config.json"

    echo "[Verify] Checking blockwise threshold config..."
    "${PY}" dit_s3cache/stage2/verify_stage2_dit.py \
        --threshold-config "${OUT_DIR}/01_blockwise_threshold/stage2_thresholds_blockwise.json"

    echo ""
    echo "Done: ${SRC} → ${OUT_DIR}"
    echo ""
done

echo "=========================================="
echo "All Stage 2 experiments complete."
echo "Results in: ${STAGE2_ROOT}"
echo "=========================================="
