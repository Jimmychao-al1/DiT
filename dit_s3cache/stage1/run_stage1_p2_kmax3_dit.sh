#!/usr/bin/env bash
# Stage-1：僅 P2 kmax3（等同 ONLY_P=2）
# 建議改用：bash dit_s3cache/stage1/run_stage1_kmax3_all_dit.sh
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ONLY_P=2 exec bash dit_s3cache/stage1/run_stage1_kmax3_all_dit.sh
