#!/usr/bin/env bash
set -euo pipefail

# Sequential DDIM block-level pipeline sweep:
#   Evidence -> Stage 0 -> Stage 1 -> Stage 2 global refine -> blockwise thresholds -> blockwise refine
#
# Usage:
#   bash dit_s3cache/scripts/run_ddim_stage2_sweep.sh
#   bash dit_s3cache/scripts/run_ddim_stage2_sweep.sh 50 100
#
# Environment overrides:
#   CONDA_ENV=ldm PY=python
#   PER_SIDE_BATCH_SIZE=4 N_BATCHES=16 CFG_SCALE=1.5 K_SVD=16 BASE_SEED=42
#   STAGE1_K=20 STAGE1_SW=3 STAGE1_LAMBDA=1.0 STAGE1_K_MAX=3
#   STAGE2_EVAL_NUM_IMAGES=8 STAGE2_EVAL_CHUNK_SIZE=1 STAGE2_SEED=42

cd "$(dirname "$0")/../.."

CONDA_ENV="${CONDA_ENV:-ldm}"
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PY:-python}"

if [ "$#" -gt 0 ]; then
  STEPS_LIST=("$@")
else
  STEPS_LIST=(50 100 150 200 250)
fi

PER_SIDE_BATCH_SIZE="${PER_SIDE_BATCH_SIZE:-4}"
N_BATCHES="${N_BATCHES:-16}"
CFG_SCALE="${CFG_SCALE:-1.5}"
K_SVD="${K_SVD:-16}"
BASE_SEED="${BASE_SEED:-42}"
ETA="${ETA:-0.0}"

STAGE1_K="${STAGE1_K:-20}"
STAGE1_SW="${STAGE1_SW:-3}"
STAGE1_LAMBDA="${STAGE1_LAMBDA:-1.0}"
STAGE1_K_MIN="${STAGE1_K_MIN:-1}"
STAGE1_K_MAX="${STAGE1_K_MAX:-3}"
STAGE1_MIN_ZONE_LEN="${STAGE1_MIN_ZONE_LEN:-2}"

STAGE2_EVAL_NUM_IMAGES="${STAGE2_EVAL_NUM_IMAGES:-8}"
STAGE2_EVAL_CHUNK_SIZE="${STAGE2_EVAL_CHUNK_SIZE:-1}"
STAGE2_SEED="${STAGE2_SEED:-42}"
Q_ZONE="${Q_ZONE:-0.90}"
Q_PEAK="${Q_PEAK:-0.80}"
PEAK_OVER_ZONE_RATIO_MIN="${PEAK_OVER_ZONE_RATIO_MIN:-1.3}"

SOURCE_NAME="baseline_p2_K${STAGE1_K}_sw${STAGE1_SW}_lam${STAGE1_LAMBDA}_kmax${STAGE1_K_MAX}"

require_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "[ERROR] Expected file missing: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [ ! -d "${path}" ]; then
    echo "[ERROR] Expected directory missing: ${path}" >&2
    exit 1
  fi
}

echo "=== DDIM DiT S3-Cache block-level sweep ==="
echo "Steps: ${STEPS_LIST[*]}"
echo "Source name: ${SOURCE_NAME}"
echo "Conda env: ${CONDA_ENV}"
echo

for STEPS in "${STEPS_LIST[@]}"; do
  echo "============================================================"
  echo "DDIM ${STEPS}-step pipeline start"
  echo "============================================================"

  EVIDENCE_NPZ="dit_s3cache/outputs/evidence_dit_xl2_256_ddim${STEPS}.npz"
  STAGE0_DIR="dit_s3cache/stage0/stage0_output_ddim${STEPS}"
  STAGE1_DIR="dit_s3cache/stage1/stage1_output_ddim${STEPS}/${SOURCE_NAME}"
  STAGE1_JSON="${STAGE1_DIR}/scheduler_config.json"
  STAGE2_DIR="dit_s3cache/stage2/stage2_output_ddim${STEPS}/src_${SOURCE_NAME}"

  echo
  echo "[1/8] Evidence collection -> ${EVIDENCE_NPZ}"
  "${PY}" -m dit_s3cache.evidence.collect_evidence \
    --sampler ddim --eta "${ETA}" \
    --num-sampling-steps "${STEPS}" \
    --per-side-batch-size "${PER_SIDE_BATCH_SIZE}" \
    --n-batches "${N_BATCHES}" \
    --cfg-scale "${CFG_SCALE}" \
    --k-svd "${K_SVD}" \
    --base-seed "${BASE_SEED}" \
    --output "${EVIDENCE_NPZ}"
  require_file "${EVIDENCE_NPZ}"
  "${PY}" -c "import json, numpy as np; p='${EVIDENCE_NPZ}'; d=np.load(p, allow_pickle=True); m=json.loads(str(d['metadata_json'])); assert m['sampler']=='ddim'; assert int(m['num_sampling_steps'])==${STEPS}; assert int(m['n_steps_recorded'])==${STEPS}; print('[check] evidence metadata OK:', p)"

  echo
  echo "[2/8] Stage 0 normalization -> ${STAGE0_DIR}"
  "${PY}" -m dit_s3cache.stage0.stage0_dit \
    --evidence-npz "${EVIDENCE_NPZ}" \
    --fid-json dit_s3cache/fid/fid_sensitivity_results.json \
    --output-dir "${STAGE0_DIR}" \
    --n-steps "${STEPS}"
  require_file "${STAGE0_DIR}/stage0_metadata.json"
  "${PY}" -c "import json, numpy as np; p='${STAGE0_DIR}'; m=json.load(open(p+'/stage0_metadata.json')); assert m['sampler']=='ddim'; assert int(m['n_steps'])==${STEPS}; assert np.load(p+'/l1_interval_norm.npy').shape==(28, ${STEPS}-1); print('[check] Stage0 outputs OK:', p)"

  echo
  echo "[3/8] Stage 1 synthesis -> ${STAGE1_DIR}"
  "${PY}" -m dit_s3cache.stage1.stage1_scheduler_dit \
    --stage0_dir "${STAGE0_DIR}" \
    --output_dir "${STAGE1_DIR}" \
    --K "${STAGE1_K}" \
    --smooth_window "${STAGE1_SW}" \
    --lambda "${STAGE1_LAMBDA}" \
    --k_min "${STAGE1_K_MIN}" \
    --k_max "${STAGE1_K_MAX}" \
    --min_zone_len "${STAGE1_MIN_ZONE_LEN}"
  require_file "${STAGE1_JSON}"

  echo
  echo "[4/8] Verify Stage 1 scheduler"
  "${PY}" -m dit_s3cache.stage1.verify_scheduler_dit --config "${STAGE1_JSON}"

  mkdir -p "${STAGE2_DIR}/00_global_refine" "${STAGE2_DIR}/01_blockwise_threshold" "${STAGE2_DIR}/02_refined_blockwise"

  echo
  echo "[5/8] Stage 2 pass 1: global refine -> ${STAGE2_DIR}/00_global_refine"
  "${PY}" -m dit_s3cache.stage2.stage2_runtime_refine_dit \
    --scheduler-config "${STAGE1_JSON}" \
    --threshold-mode global \
    --output-dir "${STAGE2_DIR}/00_global_refine" \
    --sampler ddim --eta "${ETA}" \
    --num-sampling-steps "${STEPS}" \
    --eval-num-images "${STAGE2_EVAL_NUM_IMAGES}" \
    --eval-chunk-size "${STAGE2_EVAL_CHUNK_SIZE}" \
    --seed "${STAGE2_SEED}"
  require_file "${STAGE2_DIR}/00_global_refine/stage2_runtime_diagnostics.json"

  echo
  echo "[6/8] Build blockwise thresholds -> ${STAGE2_DIR}/01_blockwise_threshold"
  "${PY}" -m dit_s3cache.stage2.build_blockwise_thresholds_dit \
    --diagnostics "${STAGE2_DIR}/00_global_refine/stage2_runtime_diagnostics.json" \
    --output "${STAGE2_DIR}/01_blockwise_threshold/stage2_thresholds_blockwise.json" \
    --q-zone "${Q_ZONE}" \
    --q-peak "${Q_PEAK}" \
    --peak-over-zone-ratio-min "${PEAK_OVER_ZONE_RATIO_MIN}" \
    --source "ddim${STEPS}_${SOURCE_NAME}"
  require_file "${STAGE2_DIR}/01_blockwise_threshold/stage2_thresholds_blockwise.json"

  echo
  echo "[7/8] Stage 2 pass 2: blockwise refine -> ${STAGE2_DIR}/02_refined_blockwise"
  "${PY}" -m dit_s3cache.stage2.stage2_runtime_refine_dit \
    --scheduler-config "${STAGE1_JSON}" \
    --threshold-mode blockwise \
    --threshold-config "${STAGE2_DIR}/01_blockwise_threshold/stage2_thresholds_blockwise.json" \
    --output-dir "${STAGE2_DIR}/02_refined_blockwise" \
    --sampler ddim --eta "${ETA}" \
    --num-sampling-steps "${STEPS}" \
    --eval-num-images "${STAGE2_EVAL_NUM_IMAGES}" \
    --eval-chunk-size "${STAGE2_EVAL_CHUNK_SIZE}" \
    --seed "${STAGE2_SEED}"
  require_file "${STAGE2_DIR}/02_refined_blockwise/stage2_refined_scheduler_config.json"

  echo
  echo "[8/8] Verify Stage 2 outputs"
  "${PY}" -m dit_s3cache.stage2.verify_stage2_dit \
    "${STAGE2_DIR}/02_refined_blockwise/stage2_refined_scheduler_config.json"
  "${PY}" -m dit_s3cache.stage2.verify_stage2_dit \
    --threshold-config "${STAGE2_DIR}/01_blockwise_threshold/stage2_thresholds_blockwise.json"

  require_dir "${STAGE2_DIR}/02_refined_blockwise"
  echo
  echo "DDIM ${STEPS}-step pipeline complete:"
  echo "  Evidence: ${EVIDENCE_NPZ}"
  echo "  Stage 0:  ${STAGE0_DIR}"
  echo "  Stage 1:  ${STAGE1_DIR}"
  echo "  Stage 2:  ${STAGE2_DIR}"
  echo
done

echo "============================================================"
echo "All DDIM step pipelines complete."
echo "============================================================"
