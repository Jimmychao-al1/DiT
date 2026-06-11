#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Q-DiT W8A8 Stage 4
# Part 1: Stage 0 evidence (DDIM 50/100/150)
# Part 2: D1 baseline FID@5K (DDIM 50/100/150)
# ============================================================

REPO="/home/jimmy/DiT"
QDIT_CKPT="/home/jimmy/Q-DiT/results/003-qdit_w8a8/qdit_w8a8_state_dict.pt"
FID_JSON="${REPO}/dit_s3cache/fid/fid_sensitivity_results.json"
RESULT_DIR="/home/jimmy/Q-DiT/results/003-qdit_w8a8"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"

cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ldm
export PYTHONPATH="${REPO}"
export PYTHONUNBUFFERED=1
mkdir -p "${RESULT_DIR}" "${REPO}/dit_s3cache/outputs"

log_ts() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

preflight_checks() {
  log_ts "Preflight checks..."
  test -f "${QDIT_CKPT}" || { log_ts "[ERROR] checkpoint not found: ${QDIT_CKPT}"; exit 1; }
  test -f "${FID_JSON}" || { log_ts "[ERROR] fid json not found: ${FID_JSON}"; exit 1; }

  python -u - <<'PY'
import shutil, torch
from pathlib import Path

ckpt = Path("/home/jimmy/Q-DiT/results/003-qdit_w8a8/qdit_w8a8_state_dict.pt")
print(f"[OK] checkpoint size: {ckpt.stat().st_size / 1e9:.2f} GB")
print(f"[OK] torch: {torch.__version__}")
print(f"[OK] cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"[OK] GPU {i}: {name} | free {free/1e9:.1f} GB / total {total/1e9:.1f} GB")
PY

  if command -v nvidia-smi >/dev/null 2>&1; then
    log_ts "nvidia-smi snapshot:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  fi

  local swap_total swap_used
  read -r swap_total swap_used _ < <(free -m | awk '/^Swap:/ {print $2, $3, $4}')
  if [[ "${swap_total}" -gt 0 ]]; then
    local swap_pct=$(( swap_used * 100 / swap_total ))
    log_ts "Swap: ${swap_used}MB / ${swap_total}MB (${swap_pct}%)"
    if [[ "${swap_pct}" -ge 90 ]]; then
      log_ts "[WARN] Swap 幾乎滿了，載入模型可能很慢"
    fi
  fi
}

run_with_heartbeat() {
  local label="$1"
  local logfile="$2"
  shift 2

  : > "${logfile}"
  local start_ts
  start_ts=$(date +%s)
  log_ts "START ${label}"
  log_ts "Command: $*"
  log_ts "Log: ${logfile}"
  log_ts "Heartbeat every ${HEARTBEAT_SEC}s (看 log 大小 / GPU 是否在動)"

  (
    local last_size=0
    local stall_count=0
    while true; do
      sleep "${HEARTBEAT_SEC}"
      local now elapsed size gpu_line py_line last_line
      now=$(date +%s)
      elapsed=$(( now - start_ts ))
      size=$(stat -c%s "${logfile}" 2>/dev/null || echo 0)

      if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_line=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "n/a")
      else
        gpu_line="nvidia-smi unavailable"
      fi

      py_line=$(pgrep -af "collect_evidence_qdit|fid_cache_sensitivity_qdit" 2>/dev/null | head -1 || true)
      if [[ -z "${py_line}" ]]; then
        py_line="(no matching python process visible)"
      fi

      if [[ -s "${logfile}" ]]; then
        last_line=$(tail -n 1 "${logfile}" | sed 's/^[[:space:]]*//')
      else
        last_line="(log still empty — likely model loading, first batch not done yet)"
      fi

      log_ts "[HEARTBEAT] ${label} | elapsed ${elapsed}s | log ${size} bytes"
      log_ts "[HEARTBEAT] GPU: ${gpu_line}"
      log_ts "[HEARTBEAT] PROC: ${py_line}"
      log_ts "[HEARTBEAT] LAST: ${last_line}"

      if [[ "${size}" -eq "${last_size}" ]]; then
        stall_count=$(( stall_count + 1 ))
        if [[ "${stall_count}" -ge 10 ]]; then
          log_ts "[WARN] log 已 ${stall_count} 個 heartbeat 沒有變大，可能卡住（或仍在載入模型）"
        fi
      else
        stall_count=0
        last_size="${size}"
      fi
    done
  ) &
  local hb_pid=$!

  set +e
  stdbuf -oL -eL "$@" 2>&1 | stdbuf -oL -eL tee -a "${logfile}"
  local cmd_status=${PIPESTATUS[0]}
  set -e

  kill "${hb_pid}" 2>/dev/null || true
  wait "${hb_pid}" 2>/dev/null || true

  local end_ts elapsed
  end_ts=$(date +%s)
  elapsed=$(( end_ts - start_ts ))
  if [[ "${cmd_status}" -ne 0 ]]; then
    log_ts "[ERROR] ${label} failed (exit=${cmd_status}) after ${elapsed}s"
    log_ts "Tail of log:"
    tail -n 30 "${logfile}" || true
    exit "${cmd_status}"
  fi
  log_ts "DONE ${label} in ${elapsed}s"
}

run_evidence() {
  local steps=$1
  local npz="${REPO}/dit_s3cache/outputs/evidence_qdit_w8a8_ddim${steps}.npz"
  local log="${REPO}/dit_s3cache/outputs/evidence_qdit_w8a8_ddim${steps}.log"
  local outdir="${REPO}/dit_s3cache/stage0/stage0_output_qdit_ddim${steps}"

  echo
  echo "============================================================"
  log_ts "[Part 1] Evidence + Stage0 | DDIM ${steps}"
  echo "============================================================"

  run_with_heartbeat "evidence_ddim${steps}" "${log}" \
    python -u dit_s3cache/evidence/collect_evidence_qdit.py \
      --num-sampling-steps "${steps}" \
      --sampler ddim \
      --cfg-scale 1.5 \
      --n-batches 16 \
      --per-side-batch-size 4 \
      --base-seed 42 \
      --qdit-ckpt "${QDIT_CKPT}" \
      --output "${npz}"

  python -u - <<PY
import numpy as np
steps = ${steps}
d = np.load("${npz}", allow_pickle=True)
for k in ["l1_diff", "cos_sim", "grassmann_dist"]:
    a = d[k]
    assert a.shape == (28, steps), f"{k}: {a.shape}"
    assert a[:, 1:].shape == (28, steps - 1)
    assert not np.isinf(a).any(), f"{k} has inf"
print(f"[OK] evidence DDIM {steps}: interval shape (28, {steps - 1})")
PY

  log_ts "Running stage0_dit.py for DDIM ${steps}..."
  mkdir -p "${outdir}"
  python -u dit_s3cache/stage0/stage0_dit.py \
    --evidence-npz "${npz}" \
    --fid-json "${FID_JSON}" \
    --output-dir "${outdir}" \
    --n-steps "${steps}"

  cp "${outdir}/fid_w_qdiffae_clip.npy" "${outdir}/fid_w_qdit_clip.npy"

  for f in l1_interval_norm.npy cosdist_interval_norm.npy svd_interval_norm.npy \
           fid_w_qdit_clip.npy block_names.npy stage0_metadata.json; do
    test -f "${outdir}/${f}" || { log_ts "[ERROR] missing ${outdir}/${f}"; exit 1; }
  done
  log_ts "[OK] stage0_output_qdit_ddim${steps}"
}

run_d1_fid() {
  local steps=$1
  local json="${REPO}/dit_s3cache/fid/d1_baseline_fid_qdit_ddim${steps}_5k.json"
  local log="${RESULT_DIR}/d1_fid_ddim${steps}.txt"

  echo
  echo "============================================================"
  log_ts "[Part 2] D1 Baseline FID@5K | DDIM ${steps}"
  echo "============================================================"

  run_with_heartbeat "d1_fid_ddim${steps}" "${log}" \
    python -u dit_s3cache/fid/fid_cache_sensitivity_qdit.py \
      --baseline-only \
      --num-sampling-steps "${steps}" \
      --sampler ddim \
      --cfg-scale 1.5 \
      --num-fid-samples 5000 \
      --per-side-batch-size 32 \
      --qdit-ckpt "${QDIT_CKPT}" \
      --results-json "${json}"

  test -f "${json}" || { log_ts "[ERROR] missing ${json}"; exit 1; }
  log_ts "[OK] saved ${json}"
}

echo "============================================================"
log_ts "START pipeline: evidence (50/100/150) -> FID@5K (50/100/150)"
echo "============================================================"
preflight_checks

for steps in 50 100 150; do
  run_evidence "${steps}"
done

echo
echo "============================================================"
log_ts "Part 1 DONE. Starting Part 2..."
echo "============================================================"

for steps in 50 100 150; do
  run_d1_fid "${steps}"
done

python -u - <<'PY'
import json, re
from datetime import datetime
from pathlib import Path

repo = Path("/home/jimmy/DiT")
out = Path("/home/jimmy/Q-DiT/results/003-qdit_w8a8/d1_fid_summary.json")

def parse(jpath: Path):
    text = json.loads(jpath.read_text())["results"]["baseline_meta"]["adm_output"]
    def pick(pat, name):
        m = re.search(pat, text)
        if not m:
            raise RuntimeError(f"cannot parse {name} from {jpath}")
        return float(m.group(1))
    return {
        "FID": pick(r"\bFID:\s*([0-9eE+\-.]+)", "FID"),
        "sFID": pick(r"\bsFID:\s*([0-9eE+\-.]+)", "sFID"),
        "IS": pick(r"\bInception Score:\s*([0-9eE+\-.]+)", "IS"),
        "Precision": pick(r"\bPrecision:\s*([0-9eE+\-.]+)", "Precision"),
        "Recall": pick(r"\bRecall:\s*([0-9eE+\-.]+)", "Recall"),
    }

summary = {
    "experiment": "D1_QDiT_W8A8_baseline",
    "model": "DiT-XL/2",
    "quantization": "W8A8_PTQ_max_group128",
    "sampler": "DDIM",
    "cfg_scale": 1.5,
    "num_samples": 5000,
    "reference": "VIRTUAL_imagenet256_labeled.npz",
    "results": {},
    "fp_baseline_reference": {
        "ddim150_FID_50K": 2.202,
        "note": "FP DiT-XL/2 DDIM 150 步 FID@50K，供對照"
    },
    "timestamp": datetime.now().astimezone().isoformat(),
}

for steps, key in [(50, "ddim50"), (100, "ddim100"), (150, "ddim150")]:
    j = repo / f"dit_s3cache/fid/d1_baseline_fid_qdit_ddim{steps}_5k.json"
    summary["results"][key] = {"steps": steps, **parse(j)}

out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"[OK] saved {out}")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo
echo "============================================================"
log_ts "ALL DONE"
echo "============================================================"
