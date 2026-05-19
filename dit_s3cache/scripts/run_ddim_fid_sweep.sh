#!/usr/bin/env bash
set -euo pipefail

# Sequential DDIM FID sweep.
#
# Requested order:
#   1. DiT baseline FID@5K:  50 -> 100 -> 150
#   2. S3-Cache FID@5K:      50 -> 100 -> 150
#   3. DiT baseline FID@50K: 50 -> 100
#   4. S3-Cache FID@50K:     50 -> 100
#
# Usage:
#   bash dit_s3cache/scripts/run_ddim_fid_sweep.sh
#
# Environment overrides:
#   CONDA_ENV=ldm PY=python
#   ETA=0.0 PER_SIDE_BATCH_SIZE=32 CFG_SCALE=1.5 SEED=0
#   ADM_PYTHON=/path/to/python

cd "$(dirname "$0")/../.."

CONDA_ENV="${CONDA_ENV:-ldm}"
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PY:-python}"

ETA="${ETA:-0.0}"
PER_SIDE_BATCH_SIZE="${PER_SIDE_BATCH_SIZE:-32}"
CFG_SCALE="${CFG_SCALE:-1.5}"
SEED="${SEED:-0}"
VAE="${VAE:-mse}"
SOURCE_NAME="${SOURCE_NAME:-baseline_p2_K20_sw3_lam1.0_kmax3}"
CACHE_JOB_PREFIX="${CACHE_JOB_PREFIX:-baseline_p2_K20_sw3_lam1.0_kmax3_blockwise}"

SAMPLE_MODULE="dit_s3cache.start_run.sample_stage2_cache_scheduler_dit"
BASELINE_5K_STEPS=(50 100 150)
CACHE_5K_STEPS=(50 100 150)
BASELINE_50K_STEPS=(50 100)
CACHE_50K_STEPS=(50 100)

COMMON_ARGS=(
  --sampler ddim
  --eta "${ETA}"
  --per-side-batch-size "${PER_SIDE_BATCH_SIZE}"
  --cfg-scale "${CFG_SCALE}"
  --seed "${SEED}"
  --vae "${VAE}"
)

if [ -n "${ADM_PYTHON:-}" ]; then
  COMMON_ARGS+=(--adm-python "${ADM_PYTHON}")
fi

results_root_for_step() {
  local steps="$1"
  printf 'dit_s3cache/results/fid_dit_stage2_ddim%s' "${steps}"
}

scheduler_json_for_step() {
  local steps="$1"
  printf 'dit_s3cache/stage2/stage2_output_ddim%s/src_%s/02_refined_blockwise/stage2_refined_scheduler_config.json' \
    "${steps}" "${SOURCE_NAME}"
}

require_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 1
  fi
}

cleanup_scratch() {
  local steps="$1"
  local root
  root="$(results_root_for_step "${steps}")"
  if [ ! -d "${root}" ]; then
    return
  fi
  echo "[cleanup] Removing scratch files under ${root}"
  find "${root}" -type f -name '_scratch_samples.npz' -delete
  find "${root}" -type d -name '_scratch_gen_png' -prune -exec rm -rf {} +
}

run_baseline() {
  local samples="$1"
  local steps="$2"
  local root
  root="$(results_root_for_step "${steps}")"

  echo
  echo "----------------------------------------------------------------"
  echo "DiT baseline FID@${samples} | DDIM ${steps}"
  echo "results_root=${root}"
  echo "----------------------------------------------------------------"

  "${PY}" -m "${SAMPLE_MODULE}" \
    --baseline-only \
    --num-sampling-steps "${steps}" \
    --num-fid-samples "${samples}" \
    --results-root "${root}" \
    "${COMMON_ARGS[@]}"

  cleanup_scratch "${steps}"
}

run_cache() {
  local samples="$1"
  local steps="$2"
  local root json job_name
  root="$(results_root_for_step "${steps}")"
  json="$(scheduler_json_for_step "${steps}")"
  job_name="${CACHE_JOB_PREFIX}_ddim${steps}"
  require_file "${json}"

  echo
  echo "----------------------------------------------------------------"
  echo "S3-Cache FID@${samples} | DDIM ${steps}"
  echo "scheduler=${json}"
  echo "results_root=${root}"
  echo "----------------------------------------------------------------"

  "${PY}" -m "${SAMPLE_MODULE}" \
    --num-sampling-steps "${steps}" \
    --num-fid-samples "${samples}" \
    --results-root "${root}" \
    --job "${job_name}" "${json}" \
    "${COMMON_ARGS[@]}"

  cleanup_scratch "${steps}"
}

echo "============================================================"
echo "DDIM FID sweep start"
echo "Conda env: ${CONDA_ENV}"
echo "Baseline 5K steps:  ${BASELINE_5K_STEPS[*]}"
echo "Cache 5K steps:     ${CACHE_5K_STEPS[*]}"
echo "Baseline 50K steps: ${BASELINE_50K_STEPS[*]}"
echo "Cache 50K steps:    ${CACHE_50K_STEPS[*]}"
echo "============================================================"

for steps in "${BASELINE_5K_STEPS[@]}"; do
  run_baseline 5000 "${steps}"
done

for steps in "${CACHE_5K_STEPS[@]}"; do
  run_cache 5000 "${steps}"
done

for steps in "${BASELINE_50K_STEPS[@]}"; do
  run_baseline 50000 "${steps}"
done

for steps in "${CACHE_50K_STEPS[@]}"; do
  run_cache 50000 "${steps}"
done

echo
echo "============================================================"
echo "DDIM FID sweep complete"
echo "Results roots:"
for steps in 50 100 150; do
  echo "  $(results_root_for_step "${steps}")"
done
echo "============================================================"
