#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Q-DiT W8A8: Full tri-evidence -> Stage0 -> D1 FID@50K
# ============================================================

REPO="/home/jimmy/DiT"
QDIT_CKPT="/home/jimmy/Q-DiT/results/003-qdit_w8a8/qdit_w8a8_state_dict.pt"

EVIDENCE_NPZ="${REPO}/dit_s3cache/outputs/evidence_qdit_w8a8_ddim250.npz"
EVIDENCE_LOG="${REPO}/dit_s3cache/outputs/evidence_qdit_w8a8_ddim250.log"
STAGE0_DIR="${REPO}/dit_s3cache/stage0/stage0_output_qdit"
D1_JSON="${REPO}/dit_s3cache/fid/d1_baseline_fid_qdit_ddim250_50k.json"
D1_LOG="/home/jimmy/Q-DiT/d1_fid_log.txt"
D1_RESULT="/home/jimmy/Q-DiT/results/003-qdit_w8a8/d1_fid_results.json"

cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ldm
export PYTHONPATH="${REPO}"

echo "============================================================"
echo "[0] Kill old D1 background process (if any)"
echo "============================================================"
kill 2660004 2>/dev/null || true

echo
echo "============================================================"
echo "[1/4] Full tri-evidence (DDIM 250, 16 batches)"
echo "============================================================"
python dit_s3cache/evidence/collect_evidence_qdit.py \
  --num-sampling-steps 250 \
  --sampler ddim \
  --cfg-scale 1.5 \
  --n-batches 16 \
  --per-side-batch-size 4 \
  --base-seed 42 \
  --qdit-ckpt "${QDIT_CKPT}" \
  --output "${EVIDENCE_NPZ}" \
  2>&1 | tee "${EVIDENCE_LOG}"

python - <<'PY'
import numpy as np
from pathlib import Path

npz = Path("/home/jimmy/DiT/dit_s3cache/outputs/evidence_qdit_w8a8_ddim250.npz")
d = np.load(npz, allow_pickle=True)
for k in ["l1_diff", "cos_sim", "grassmann_dist"]:
    a = d[k]
    assert a.shape == (28, 250), f"{k} shape mismatch: {a.shape}"
    assert a[:, 1:].shape == (28, 249), f"{k} interval shape mismatch"
    assert not np.isinf(a).any(), f"{k} contains inf"
print("[OK] evidence shape check passed")
PY

echo
echo "============================================================"
echo "[2/4] Stage 0 post-processing"
echo "============================================================"
mkdir -p "${STAGE0_DIR}"

python dit_s3cache/stage0/stage0_dit.py \
  --evidence-npz "${EVIDENCE_NPZ}" \
  --fid-json "${REPO}/dit_s3cache/fid/fid_sensitivity_results.json" \
  --output-dir "${STAGE0_DIR}" \
  --n-steps 250

cp "${STAGE0_DIR}/fid_w_qdiffae_clip.npy" \
   "${STAGE0_DIR}/fid_w_qdit_clip.npy"

for f in l1_interval_norm.npy cosdist_interval_norm.npy svd_interval_norm.npy fid_w_qdiffae_clip.npy fid_w_qdit_clip.npy; do
  test -f "${STAGE0_DIR}/${f}" || { echo "[ERROR] missing ${STAGE0_DIR}/${f}"; exit 1; }
done
echo "[OK] stage0_output_qdit files ready"

echo
echo "============================================================"
echo "[3/4] D1 baseline FID@50K"
echo "============================================================"
python dit_s3cache/fid/fid_cache_sensitivity_qdit.py \
  --baseline-only \
  --num-sampling-steps 250 \
  --sampler ddim \
  --cfg-scale 1.5 \
  --num-fid-samples 50000 \
  --qdit-ckpt "${QDIT_CKPT}" \
  --results-json "${D1_JSON}" \
  2>&1 | tee "${D1_LOG}"

test -f "${D1_JSON}" || { echo "[ERROR] missing ${D1_JSON}"; exit 1; }
echo "[OK] D1 raw results saved: ${D1_JSON}"

echo
echo "============================================================"
echo "[4/4] Parse and save d1_fid_results.json"
echo "============================================================"
python - <<'PY'
import json, re
from datetime import datetime
from pathlib import Path

src = Path("/home/jimmy/DiT/dit_s3cache/fid/d1_baseline_fid_qdit_ddim250_50k.json")
dst = Path("/home/jimmy/Q-DiT/results/003-qdit_w8a8/d1_fid_results.json")

payload = json.loads(src.read_text())
text = payload["results"]["baseline_meta"].get("adm_output", "")

def pick(pattern, name):
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"cannot parse {name} from adm_output")
    return float(m.group(1))

result = {
    "experiment": "D1_QDiT_W8A8_baseline",
    "model": "DiT-XL/2",
    "quantization": "W8A8_PTQ_max_group128",
    "sampler": "DDIM",
    "steps": 250,
    "cfg_scale": 1.5,
    "num_samples": 50000,
    "reference": "VIRTUAL_imagenet256_labeled.npz",
    "FID": pick(r"\bFID:\s*([0-9eE+\-.]+)", "FID"),
    "sFID": pick(r"\bsFID:\s*([0-9eE+\-.]+)", "sFID"),
    "IS": pick(r"\bInception Score:\s*([0-9eE+\-.]+)", "IS"),
    "Precision": pick(r"\bPrecision:\s*([0-9eE+\-.]+)", "Precision"),
    "Recall": pick(r"\bRecall:\s*([0-9eE+\-.]+)", "Recall"),
    "timestamp": datetime.now().astimezone().isoformat(),
}

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"[OK] saved: {dst}")
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

echo
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
echo "Evidence : ${EVIDENCE_NPZ}"
echo "Stage0   : ${STAGE0_DIR}"
echo "D1 JSON  : ${D1_JSON}"
echo "D1 Result: ${D1_RESULT}"
