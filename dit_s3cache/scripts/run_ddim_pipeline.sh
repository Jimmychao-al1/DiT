#!/usr/bin/env bash
set -euo pipefail

STEPS=${1:?Usage: run_ddim_pipeline.sh <steps>}
cd "$(dirname "$0")/../.."

echo "=== DDIM ${STEPS}-step DiT S3-Cache pipeline ==="

echo "[1/2] Collect evidence"
python -m dit_s3cache.evidence.collect_evidence \
  --sampler ddim --eta 0.0 \
  --num-sampling-steps "${STEPS}" \
  --per-side-batch-size 4 --n-batches 16 \
  --cfg-scale 1.5 --k-svd 16 --base-seed 42 \
  --output "dit_s3cache/outputs/evidence_dit_xl2_256_ddim${STEPS}.npz"

echo "[2/2] Stage 0 normalization (reuse static FID sensitivity)"
python -m dit_s3cache.stage0.stage0_dit \
  --evidence-npz "dit_s3cache/outputs/evidence_dit_xl2_256_ddim${STEPS}.npz" \
  --fid-json dit_s3cache/fid/fid_sensitivity_results.json \
  --output-dir "dit_s3cache/stage0/stage0_output_ddim${STEPS}" \
  --n-steps "${STEPS}"

cat <<EOF
=== Stage 0 complete for DDIM ${STEPS} ===

Next stages should use suffixed output roots, for example:

python -m dit_s3cache.stage1.stage1_scheduler_dit \\
  --stage0_dir dit_s3cache/stage0/stage0_output_ddim${STEPS} \\
  --output_dir dit_s3cache/stage1/stage1_output_ddim${STEPS}/baseline_p2_K20_sw3_lam1.0_kmax3 \\
  --K 20 --smooth_window 3 --lambda 1.0 --k_max 3

python -m dit_s3cache.stage2.stage2_runtime_refine_dit \\
  --scheduler-config dit_s3cache/stage1/stage1_output_ddim${STEPS}/baseline_p2_K20_sw3_lam1.0_kmax3/scheduler_config.json \\
  --output-dir dit_s3cache/stage2/stage2_output_ddim${STEPS}/src_baseline_p2_K20_sw3_lam1.0_kmax3/00_global_refine \\
  --sampler ddim --eta 0.0 --num-sampling-steps ${STEPS}

python -m dit_s3cache.start_run.sample_stage2_cache_scheduler_dit \\
  --sampler ddim --eta 0.0 --num-sampling-steps ${STEPS} \\
  --job baseline_p2_K20_sw3_lam1.0_kmax3_blockwise <stage2-refined-json>
EOF
