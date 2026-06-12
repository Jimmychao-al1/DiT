#!/usr/bin/env bash
set -euo pipefail

REPO="/home/jimmy/DiT"
QDIT_CKPT="/home/jimmy/Q-DiT/results/003-qdit_w8a8/qdit_w8a8_state_dict.pt"
PER_SIDE_BATCH_SIZE="${PER_SIDE_BATCH_SIZE:-16}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-5000}"
SEED="${SEED:-0}"

cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ldm
export PYTHONPATH="${REPO}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${QDIT_CKPT}" ]]; then
  echo "[ERROR] Q-DiT checkpoint not found: ${QDIT_CKPT}"
  exit 1
fi

run_t() {
  local T="$1"
  local sweep_src="$2"
  local sweep_tag="$3"

  local stage2_root="dit_s3cache/stage2/stage2_output_qdit_ddim${T}"
  local results_root="dit_s3cache/results/fid_qdit_stage2_ddim${T}"

  local baseline_json="${stage2_root}/src_baseline_p2_K20_sw3_lam1.0_kmax3/02_refined_blockwise/stage2_refined_scheduler_config.json"
  local sweep_json="${stage2_root}/src_${sweep_src}/02_refined_blockwise/stage2_refined_scheduler_config.json"

  for f in "${baseline_json}" "${sweep_json}"; do
    if [[ ! -f "${f}" ]]; then
      echo "[ERROR] missing scheduler JSON: ${f}"
      exit 1
    fi
  done

  echo "============================================================"
  echo "Q-DiT start_run | DDIM T=${T} | FID@${NUM_FID_SAMPLES}"
  echo "  baseline: ${baseline_json}"
  echo "  sweep:    ${sweep_json}"
  echo "  results:  ${results_root}"
  echo "============================================================"

  python -u dit_s3cache/start_run/sample_stage2_cache_scheduler_dit.py \
    --qdit-ckpt "${QDIT_CKPT}" \
    --sampler ddim \
    --eta 0.0 \
    --num-sampling-steps "${T}" \
    --num-fid-samples "${NUM_FID_SAMPLES}" \
    --per-side-batch-size "${PER_SIDE_BATCH_SIZE}" \
    --cfg-scale 1.5 \
    --seed "${SEED}" \
    --results-root "${results_root}" \
    --job "qdit_baseline_p2_K20_sw3_blockwise_ddim${T}" "${baseline_json}" \
    --job "qdit_${sweep_tag}_blockwise_ddim${T}" "${sweep_json}"

  echo "[OK] T=${T} done"
  echo
}

# T=50, sweep winner K8_sw3
# T=100, sweep winner K8_sw3
# T=150, sweep winner K15_sw5
declare -a SPECS=(
  "50|sweep_K8_sw3_lam1.0_kmax3|sweep_K8_sw5"
  "100|sweep_K8_sw3_lam1.0_kmax3|sweep_K8_sw3"
  "150|sweep_K15_sw5_lam1.0_kmax3|sweep_K15_sw5"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r T sweep_src sweep_tag <<< "${spec}"
  if [[ -n "${ONLY_T:-}" && "${ONLY_T}" != "${T}" ]]; then
    continue
  fi
  run_t "${T}" "${sweep_src}" "${sweep_tag}"
done

echo "============================================================"
echo "ALL DONE"
echo "Results under: dit_s3cache/results/fid_qdit_stage2_ddim{50,100,150}/"
echo "============================================================"
