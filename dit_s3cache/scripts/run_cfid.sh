#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PART="${1:-}"
SCRIPT="dit_s3cache/fid/fid_cache_sensitivity.py"
LOG_DIR="dit_s3cache/fid"

# 預設使用本 repo 內的 ADM 風格 evaluator（需已安裝 TensorFlow 1.x compat，見 evaluator.py 開頭）
ADM_EVALUATOR="${ADM_EVALUATOR:-$PWD/dit_s3cache/fid/evaluator.py}"
# 與 fid_cache_sensitivity.py 預設一致：dit_s3cache/fid/VIRTUAL_imagenet256_labeled.npz
REF_BATCH="${REF_BATCH:-$PWD/dit_s3cache/fid/VIRTUAL_imagenet256_labeled.npz}"

if [[ ! -f "$ADM_EVALUATOR" ]]; then
  echo "ADM evaluator not found: $ADM_EVALUATOR"
  echo "Set ADM_EVALUATOR to your evaluator.py path, or keep dit_s3cache/fid/evaluator.py in place."
  exit 1
fi

if [[ ! -f "$REF_BATCH" ]]; then
  echo "Reference batch not found: $REF_BATCH"
  echo "Place VIRTUAL_imagenet256_labeled.npz under dit_s3cache/fid/, or set REF_BATCH to its path."
  echo "TensorFlow: evaluator 使用 tensorflow.compat.v1；若尚未安裝請在環境內安裝相容版 TensorFlow（例如 pip install tensorflow）。"
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
  echo "ADM evaluator: $ADM_EVALUATOR"
  echo "REF batch:     $REF_BATCH"
  echo "=========================================="
  python "$SCRIPT" --part "$PART" "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/cfid_part_${PART}.log"
else
  echo "=========================================="
  echo "DiT c_FID Sensitivity — Full"
  echo "ADM evaluator: $ADM_EVALUATOR"
  echo "REF batch:     $REF_BATCH"
  echo "=========================================="
  python "$SCRIPT" "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/cfid_full.log"
fi

echo "Done."
