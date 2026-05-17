#!/usr/bin/env bash
# DiT S3-Cache Stage 2: sub-layer runtime refinement pipeline
#
# Default: run the controlled-comparison mainline K20/sw3/lambda=1.
#
# Usage:
#   bash dit_s3cache/stage2/sub_layer/run_stage2_sublayer_dit.sh
#   bash dit_s3cache/stage2/sub_layer/run_stage2_sublayer_dit.sh sweep_K15_sw3_lam1
#   bash dit_s3cache/stage2/sub_layer/run_stage2_sublayer_dit.sh sweep_K10_sw3_lam1
#   bash dit_s3cache/stage2/sub_layer/run_stage2_sublayer_dit.sh all
#

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

STAGE0_DIR="dit_s3cache/stage0/stage0_output_sublayer"
STAGE1_ROOT="dit_s3cache/stage1/stage1_output_sublayer"
STAGE2_ROOT="dit_s3cache/stage2/stage2_output_sublayer"

if [ ! -d "${STAGE0_DIR}" ]; then
  echo "[ERROR] Stage0 sub-layer output dir not found: ${STAGE0_DIR}"
  exit 1
fi

if [ "${1:-}" = "all" ]; then
  SOURCES=(
    "sweep_K20_sw3_lam1"
    "sweep_K15_sw3_lam1"
    "sweep_K10_sw3_lam1"
  )
else
  SOURCES=("${1:-sweep_K20_sw3_lam1}")
fi

run_one() {
  local SRC="$1"
  local STAGE1_JSON="${STAGE1_ROOT}/${SRC}/scheduler_config.json"
  local OUT_DIR="${STAGE2_ROOT}/src_${SRC}"

  if [ ! -f "${STAGE1_JSON}" ]; then
    echo "[ERROR] Stage1 sub-layer scheduler not found: ${STAGE1_JSON}"
    echo "Available sweep dirs:"
    find "${STAGE1_ROOT}" -maxdepth 1 -type d -name 'sweep_K*' -printf '  %f\n' | sort || true
    exit 1
  fi

  mkdir -p "${OUT_DIR}/00_global_refine"
  mkdir -p "${OUT_DIR}/01_sublayerwise_threshold"
  mkdir -p "${OUT_DIR}/02_refined_sublayerwise"
  rm -f "${OUT_DIR}/stage2_scheduler_summary.csv"

  echo "=========================================="
  echo "DiT Stage 2 Sub-layer Pipeline"
  echo "Source: ${SRC}"
  echo "Stage1: ${STAGE1_JSON}"
  echo "Output: ${OUT_DIR}"
  echo "=========================================="

  # ------------------------------------------
  # Pass 1: Global refine
  # ------------------------------------------
  echo ""
  echo "[Pass 1] Global refine → ${OUT_DIR}/00_global_refine"
  "${PY}" -m dit_s3cache.stage2.sub_layer.stage2_runtime_refine_sublayer_dit \
    --scheduler-config "${STAGE1_JSON}" \
    --threshold-mode global \
    --output-dir "${OUT_DIR}/00_global_refine" \
    --eval-num-images 8 \
    --eval-chunk-size 1 \
    --seed 42

  # ------------------------------------------
  # Build sub-layer thresholds
  # ------------------------------------------
  echo ""
  echo "[Threshold] Building sub-layer thresholds → ${OUT_DIR}/01_sublayerwise_threshold"
  "${PY}" -m dit_s3cache.stage2.sub_layer.stage2_runtime_refine_sublayer_dit \
    --build-thresholds \
    --diagnostics "${OUT_DIR}/00_global_refine/stage2_runtime_diagnostics.json" \
    --output "${OUT_DIR}/01_sublayerwise_threshold/stage2_thresholds_sublayerwise.json" \
    --q-zone 0.90 \
    --q-peak 0.80 \
    --peak-over-zone-ratio-min 1.3

  # ------------------------------------------
  # Pass 2: Sub-layer-wise refine
  # ------------------------------------------
  echo ""
  echo "[Pass 2] Sub-layer-wise refine → ${OUT_DIR}/02_refined_sublayerwise"
  "${PY}" -m dit_s3cache.stage2.sub_layer.stage2_runtime_refine_sublayer_dit \
    --scheduler-config "${STAGE1_JSON}" \
    --threshold-mode sublayerwise \
    --threshold-config "${OUT_DIR}/01_sublayerwise_threshold/stage2_thresholds_sublayerwise.json" \
    --output-dir "${OUT_DIR}/02_refined_sublayerwise" \
    --eval-num-images 8 \
    --eval-chunk-size 1 \
    --seed 42

  # ------------------------------------------
  # Adapter: refined scheduler → runtime recompute schedule
  # ------------------------------------------
  echo ""
  echo "[Adapter] Building runtime recompute schedule → ${OUT_DIR}/cache_schedule_sublayer.json"
  "${PY}" -m dit_s3cache.start_run.stage2_to_dit_cache_sublayer \
    --scheduler-json "${OUT_DIR}/02_refined_sublayerwise/stage2_refined_scheduler_config.json" \
    --output-json "${OUT_DIR}/cache_schedule_sublayer.json" \
    --num-sampling-steps 250

  echo ""
  echo "✅ Stage 2 sub-layer complete: ${SRC}"
  echo "Outputs:"
  echo "  ${OUT_DIR}/00_global_refine/stage2_refined_scheduler_config.json"
  echo "  ${OUT_DIR}/01_sublayerwise_threshold/stage2_thresholds_sublayerwise.json"
  echo "  ${OUT_DIR}/02_refined_sublayerwise/stage2_refined_scheduler_config.json"
  echo "  ${OUT_DIR}/cache_schedule_sublayer.json"
  echo ""
}

for SRC in "${SOURCES[@]}"; do
  run_one "${SRC}"
done

echo "=========================================="
echo "All requested Stage 2 sub-layer runs complete."
echo "Results in: ${STAGE2_ROOT}"
echo "=========================================="
