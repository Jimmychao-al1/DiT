#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m dit_s3cache.evidence.collect_evidence \
  --image-size 256 \
  --num-sampling-steps 250 \
  --per-side-batch-size 4 \
  --n-batches 16 \
  --cfg-scale 4.0 \
  --k-svd 16 \
  --base-seed 42 \
  --output dit_s3cache/outputs/evidence_dit_xl2_256.npz
