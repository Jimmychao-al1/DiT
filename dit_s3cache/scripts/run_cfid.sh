#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PART="${1:-}"
SCRIPT="dit_s3cache/fid/fid_cache_sensitivity.py"
LOG_DIR="dit_s3cache/fid"

ADM_EVALUATOR="${ADM_EVALUATOR:-$PWD/dit_s3cache/fid/evaluator.py}"
REF_BATCH="${REF_BATCH:-$PWD/dit_s3cache/fid/VIRTUAL_imagenet256_labeled.npz}"
RESULTS_JSON="${RESULTS_JSON:-$PWD/dit_s3cache/fid/fid_sensitivity_results.json}"

# 收尾階段進度提示（實際待辦以共用 JSON 內有效 fid 為準）
PART_A_COMPLETED="${PART_A_COMPLETED:-28}"
PART_B_COMPLETED="${PART_B_COMPLETED:-32}"
# Part-A 任務數 = 28 + N_K5_ON_PART_A（舊加權預設 3 → 共 31 項；若你當時顯示 32/32 則改為 4）
N_K5_ON_PART_A="${N_K5_ON_PART_A:-3}"

if [[ ! -f "$ADM_EVALUATOR" ]]; then
  echo "ADM evaluator not found: $ADM_EVALUATOR"
  exit 1
fi

if [[ ! -f "$REF_BATCH" ]]; then
  echo "Reference batch not found: $REF_BATCH"
  exit 1
fi

COMMON_ARGS=(
  --adm-evaluator "$ADM_EVALUATOR"
  --ref-batch "$REF_BATCH"
  --results-json "$RESULTS_JSON"
  --num-fid-samples 1000
  --per-side-batch-size 32
  --cfg-scale 1.5
)

run_part() {
  local part="$1"
  shift
  echo "=========================================="
  echo "DiT c_FID — Part ${part}"
  echo "ADM evaluator: $ADM_EVALUATOR"
  echo "REF batch:     $REF_BATCH"
  echo "Results JSON:  $RESULTS_JSON"
  echo "=========================================="
  PYTHONPATH="$PWD" python "$SCRIPT" --part "$part" "$@" "${COMMON_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/cfid_part_${part}.log"
}

if [[ -n "$PART" ]]; then
  case "$PART" in
    A|B)
      run_part "$PART"
      ;;
    A2|B2)
      run_part "$PART" \
        --part-a-completed "$PART_A_COMPLETED" \
        --part-b-completed "$PART_B_COMPLETED" \
        --n-k5-on-part-a "$N_K5_ON_PART_A"
      ;;
    *)
      echo "Usage: $0 [A|B|A2|B2]"
      echo ""
      echo "  A / B   — 對稱 42+42（全新一輪）"
      echo "  A2 / B2 — 收尾舊加權清單：依共用 fid_sensitivity_results.json"
      echo "            略過已有 fid 的 (k,block)，剩餘對半分給兩台"
      echo ""
      echo "  預設（你的進度）:"
      echo "    PART_A_COMPLETED=28  N_K5_ON_PART_A=3  → Part-A 共 31 項（若曾顯示 32 請設 N_K5_ON_PART_A=4）"
      echo "    PART_B_COMPLETED=32  → Part-B 共 53 項（需 N_K5_ON_PART_A=3）"
      echo "  兩台必須共用同一份 results JSON（可用 RESULTS_JSON 指定）"
      exit 1
      ;;
  esac
else
  echo "=========================================="
  echo "DiT c_FID — Full (single host)"
  echo "=========================================="
  PYTHONPATH="$PWD" python "$SCRIPT" "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/cfid_full.log"
fi

echo "Done."
