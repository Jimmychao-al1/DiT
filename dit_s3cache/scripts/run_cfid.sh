#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PART="${1:-}"
SCRIPT="dit_s3cache/fid/fid_cache_sensitivity.py"
LOG_DIR="dit_s3cache/fid"

ADM_EVALUATOR="${ADM_EVALUATOR:-}"
REF_BATCH="${REF_BATCH:-}"

if [[ -z "$ADM_EVALUATOR" || -z "$REF_BATCH" ]]; then
  echo "Please set ADM_EVALUATOR and REF_BATCH before running."
  echo "Example:"
  echo "  ADM_EVALUATOR=/path/to/guided-diffusion/evaluations/evaluator.py \\"
  echo "  REF_BATCH=/path/to/VIRTUAL_imagenet256_labeled.npz \\"
  echo "  bash dit_s3cache/scripts/run_cfid.sh [A|B]"
  exit 1
fi

COMMON_ARGS=(
  --adm-evaluator "$ADM_EVALUATOR"
  --ref-batch "$REF_BATCH"
  --num-fid-samples 1000
  --per-side-batch-size 8
  --cfg-scale 4.0
)

if [[ -n "$PART" ]]; then
  if [[ "$PART" != "A" && "$PART" != "B" ]]; then
    echo "Usage: $0 [A|B]"
    echo "  A = baseline + k=3 all + k=5 blocks 0~13"
    echo "  B = baseline + k=5 blocks 14~27 + k=10 all"
    echo "  no arg = single-host full sweep"
    exit 1
  fi

  echo "=========================================="
  echo "DiT c_FID Sensitivity — Part ${PART}"
  echo "=========================================="
  python "$SCRIPT" --part "$PART" "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/cfid_part_${PART}.log"
else
  echo "=========================================="
  echo "DiT c_FID Sensitivity — Full"
  echo "=========================================="
  python "$SCRIPT" "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/cfid_full.log"
fi

echo "Done."
