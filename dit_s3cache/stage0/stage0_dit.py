"""
DiT S3-Cache Stage 0: Normalization + FID Weights

功能：
  讀取 DiT evidence NPZ → min-max 正規化 → 計算 FID-based block weights
  → 輸出標準格式 .npy 供 Stage 1 (load_stage0_formal) 消費。

與 Diff-AE stage0e_normalization.py 的對應：
  - 輸入：單一 NPZ (28, 250) 取代 per-block .npz + per-block .json
  - SVD key: grassmann_dist（Grassmann geodesic）取代 subspace_dist（Frobenius-based）
  - Blocks: 28 個 DiTBlock（flat）取代 31 個 UNet block
  - T=250 取代 T=100；k_values={3,5,10} 取代 {3,4,5}
  - FID block 命名 block_0..block_27 與 evidence 一致，無需 name mapping
  - 輸出 .npy 檔名與 stage0e_normalization.py 完全相同（Stage 1 依賴固定檔名）
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("Stage0DiT")

# Expected constants (verified at load time; overridable via CLI for non-standard runs)
_DEFAULT_N_BLOCKS = 28
_DEFAULT_N_STEPS = 250
_K_VALUES_DIT = [3, 5, 10]


# =============================================================================
# 一、載入 Evidence NPZ
# =============================================================================


def load_dit_evidence(
    npz_path: str,
    expected_n_blocks: int = _DEFAULT_N_BLOCKS,
    expected_n_steps: int = _DEFAULT_N_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], int, int]:
    """
    載入 DiT evidence NPZ，提取 interval-wise 指標。

    Interval 定義：
      - step_idx=0 是第一個 sampling step（無前步可比，pair metrics 為 NaN）。
      - Interval j 定義為 step_idx=j 與 step_idx=j-1 之間的變化，j ∈ [1, T-1]。
      - 因此 interval evidence = [:, 1:]，shape (B, T-1) = (28, 249)。

    Returns:
        l1_interval:     (B, T-1) L1 relative difference
        cosdist_interval:(B, T-1) cosine distance = 1 - cos_sim
        svd_interval:    (B, T-1) Grassmann geodesic distance
        evidence_meta:   dict，從 metadata_json 解析
        B:               block 數量
        T:               總 sampling step 數
    """
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evidence NPZ 不存在: {path}")

    data = np.load(path, allow_pickle=True)

    # ---- 讀取必要 arrays ----
    for key in ("l1_diff", "cos_sim", "grassmann_dist", "pair_count"):
        if key not in data:
            raise KeyError(f"Evidence NPZ 缺少必要 key: '{key}' (found: {list(data.keys())})")

    l1_full = data["l1_diff"].astype(np.float64)            # (B, T)
    cos_full = data["cos_sim"].astype(np.float64)           # (B, T)
    svd_full = data["grassmann_dist"].astype(np.float64)    # (B, T)
    pair_count = data["pair_count"]                          # (B, T)

    B, T = l1_full.shape
    LOGGER.info(f"Evidence NPZ 載入：B={B}, T={T}")

    # ---- 形狀驗證 ----
    if B != expected_n_blocks:
        LOGGER.warning(f"B={B} 與預期 {expected_n_blocks} 不符，繼續執行（可調整 --n-blocks）")
    if T != expected_n_steps:
        LOGGER.warning(f"T={T} 與預期 {expected_n_steps} 不符，繼續執行（可調整 --n-steps）")
    if cos_full.shape != (B, T) or svd_full.shape != (B, T):
        raise ValueError(
            f"Array shapes 不一致：l1={l1_full.shape}, cos={cos_full.shape}, svd={svd_full.shape}"
        )

    # ---- 驗證 step_idx=0 確實無 pair ----
    pc0 = pair_count[:, 0]
    if np.any(pc0 != 0):
        LOGGER.warning(
            f"pair_count[:, 0] 不全為 0（{np.sum(pc0 != 0)} 個 block 例外），"
            "可能 evidence 格式有變，請確認。"
        )
    else:
        LOGGER.info("✅ pair_count[:, 0] 全為 0（step_idx=0 確認無 pair）")

    # ---- 切掉 step_idx=0（NaN pair），取 intervals ----
    l1_interval = l1_full[:, 1:]          # (B, T-1)
    cosdist_interval = 1.0 - cos_full[:, 1:]  # (B, T-1)，轉成 cosine distance
    svd_interval = svd_full[:, 1:]        # (B, T-1)

    LOGGER.info(f"Interval 陣列 shape: {l1_interval.shape}  (= B × T-1)")

    # ---- 讀取 metadata ----
    evidence_meta: dict[str, Any] = {}
    if "metadata_json" in data:
        try:
            raw = data["metadata_json"]
            meta_str = str(raw) if raw.ndim == 0 else str(raw.item())
            evidence_meta = json.loads(meta_str)
        except Exception as e:
            LOGGER.warning(f"metadata_json 解析失敗（繼續執行）: {e}")

    return l1_interval, cosdist_interval, svd_interval, evidence_meta, B, T


# =============================================================================
# 二、Min-max 正規化（與 stage0e_normalization.py 完全相同）
# =============================================================================


def normalize_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    對 x 全體元素做 min-max 正規化到 [0, 1]。

    - 過濾 NaN/Inf
    - 若 max - min > eps：x_norm = (x - min) / (max - min)
    - 否則：全零
    - 最後 clip 到 [0, 1]
    """
    valid_mask = np.isfinite(x)

    if not np.any(valid_mask):
        LOGGER.warning("輸入陣列全為 NaN/Inf，回傳全零")
        return np.zeros_like(x, dtype=np.float32)

    x_valid = x[valid_mask]
    x_min = float(x_valid.min())
    x_max = float(x_valid.max())

    if x_max - x_min <= eps:
        LOGGER.warning(f"數值範圍過小 (max - min = {x_max - x_min:.2e} <= eps={eps:.2e})，回傳全零")
        return np.zeros_like(x, dtype=np.float32)

    x_norm = (x - x_min) / (x_max - x_min)
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=1.0, neginf=0.0)
    x_norm = np.clip(x_norm, 0.0, 1.0)
    return x_norm.astype(np.float32)


# =============================================================================
# 三、載入 FID Sensitivity JSON（DiT 格式）
# =============================================================================


def load_dit_delta_fid(
    fid_json_path: str,
    k_values: list[int] | None = None,
) -> tuple[list[str], dict[str, dict[int, float]], float]:
    """
    從 DiT FID sensitivity JSON 讀取 delta-FID。

    期望 JSON 結構（與 dit_s3cache/fid/fid_sensitivity_results.json 一致）：
        {
          "results": {
            "baseline_fid": 44.31,
            "k3":  {"block_0": {"fid": ..., "delta": ...}, ...},
            "k5":  {...},
            "k10": {...}
          }
        }

    Returns:
        block_names:  依 block 編號排序的名稱列表
        delta_fid:    delta_fid[block_name][k] = delta_FID（float）
        baseline_fid: baseline FID 值
    """
    if k_values is None:
        k_values = _K_VALUES_DIT

    fid_path = Path(fid_json_path)
    if not fid_path.is_file():
        raise FileNotFoundError(f"FID JSON 不存在: {fid_path}")

    with open(fid_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", payload)  # 相容根層或 results 層

    baseline_fid = results.get("baseline_fid")
    if baseline_fid is None:
        raise ValueError("FID JSON 缺少 'baseline_fid'")
    baseline_fid = float(baseline_fid)
    LOGGER.info(f"Baseline FID: {baseline_fid:.6f}")

    delta_fid: dict[str, dict[int, float]] = {}
    for k in k_values:
        k_key = f"k{k}"
        if k_key not in results:
            LOGGER.warning(f"FID JSON 中找不到 k={k}（key='{k_key}'），跳過")
            continue
        for block_key, block_data in results[k_key].items():
            if not isinstance(block_data, dict):
                continue
            delta = block_data.get("delta")
            if delta is None:
                # fallback：用 fid - baseline_fid 計算
                fid_val = block_data.get("fid")
                if fid_val is not None:
                    delta = float(fid_val) - baseline_fid
                else:
                    continue
            delta_fid.setdefault(block_key, {})[k] = float(delta)

    if not delta_fid:
        raise ValueError(
            f"FID JSON 中沒有讀取到任何有效的 delta，k_values={k_values}，"
            f"可用 keys: {list(results.keys())}"
        )

    # 依 block 編號排序（block_0 < block_1 < … < block_27）
    def _block_sort_key(name: str) -> int:
        parts = name.rsplit("_", 1)
        return int(parts[-1]) if len(parts) == 2 and parts[-1].isdigit() else 0

    block_names = sorted(delta_fid.keys(), key=_block_sort_key)
    LOGGER.info(f"✅ 成功載入 {len(block_names)} 個 block 的 delta-FID，k_values={k_values}")

    return block_names, delta_fid, baseline_fid


# =============================================================================
# 四、FID-based block weight 計算（與 stage0e_normalization.py 完全相同）
# =============================================================================


def rank_based_weights(S: np.ndarray) -> np.ndarray:
    """最小 S → w=0，最大 S → w=1，中間線性插值。"""
    B = S.shape[0]
    order = np.argsort(S)
    w_rank = np.zeros(B, dtype=np.float32)
    if B > 1:
        for i, idx in enumerate(order):
            w_rank[idx] = i / (B - 1)
    else:
        w_rank[0] = 1.0
    return w_rank


def compute_fid_weights(
    block_names: list[str],
    delta_fid: dict[str, dict[int, float]],
    k_values: list[int] | None = None,
    eps_noise: float = 0.5,
    quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算 FID-based block weights w_b。

    Steps：
        1. noise 修正：delta_pos = max(0, delta - eps_noise)
        2. worst-case 聚合：S_b = max_k delta_pos[b][k]
        3. quantile clipping：S_clip = min(S, quantile_hi)
        4. max-normalization：w_clip = S_clip / max(S_clip)
        5. rank-based（ablation）：線性排序 [0, 1]

    Returns:
        w_clip: (B,) quantile-clipped + max-normalized weights
        w_rank: (B,) rank-based weights
    """
    if k_values is None:
        k_values = _K_VALUES_DIT

    B = len(block_names)
    S = np.zeros(B, dtype=np.float32)

    for i, block_name in enumerate(block_names):
        if block_name not in delta_fid:
            LOGGER.warning(f"Block '{block_name}' 沒有 FID 資料，S_b=0")
            continue
        vals = [
            max(0.0, float(delta_fid[block_name][k]) - eps_noise)
            for k in k_values
            if k in delta_fid[block_name]
        ]
        if vals:
            S[i] = max(vals)

    LOGGER.info(
        f"S 統計 (eps_noise={eps_noise}): "
        f"min={S.min():.4f}, max={S.max():.4f}, mean={S.mean():.4f}, "
        f"n_nonzero={np.sum(S > 0)}/{B}"
    )

    if np.all(S == 0):
        LOGGER.warning("所有 S_b=0，回傳全零權重（可嘗試降低 --eps-noise）")
        return np.zeros(B, dtype=np.float32), np.zeros(B, dtype=np.float32)

    hi = float(np.quantile(S, quantile))
    S_clip = np.minimum(S, hi)
    LOGGER.info(f"Quantile clipping (q={quantile}): hi={hi:.4f}")

    h = float(S_clip.max())
    if h <= 0:
        LOGGER.warning("S_clip.max()<=0，回傳全零權重")
        w_clip = np.zeros(B, dtype=np.float32)
    else:
        w_clip = (S_clip / h).astype(np.float32)

    w_rank = rank_based_weights(S)

    LOGGER.info(
        f"w_clip: min={w_clip.min():.4f}, max={w_clip.max():.4f}, mean={w_clip.mean():.4f}"
    )
    LOGGER.info(
        f"w_rank: min={w_rank.min():.4f}, max={w_rank.max():.4f}, mean={w_rank.mean():.4f}"
    )
    return w_clip, w_rank


# =============================================================================
# 五、輸出前數值檢查
# =============================================================================


def _check_array(arr: np.ndarray, name: str) -> None:
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    in_range = bool((arr >= 0).all() and (arr <= 1).all())
    status = "✅" if (not has_nan and not has_inf and in_range) else "❌"
    LOGGER.info(
        f"  {status} {name}: shape={arr.shape}, "
        f"NaN={has_nan}, Inf={has_inf}, in[0,1]={in_range}, "
        f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"
    )
    if not in_range or has_nan or has_inf:
        LOGGER.error(f"    ❌ {name} 包含異常值，請檢查 evidence NPZ 來源！")


# =============================================================================
# 六、主入口函式
# =============================================================================


def run_stage0_dit(
    evidence_npz: str,
    fid_json: str,
    output_dir: str,
    eps_noise: float = 0.5,
    quantile: float = 0.95,
    k_values: list[int] | None = None,
) -> None:
    """
    DiT Stage 0 主流程：

        load evidence → normalize → compute FID weights → save .npy

    所有輸出 .npy 的檔名與 stage0e_normalization.py 完全一致，
    使 Stage 1 的 load_stage0_formal() 可直接讀取。
    """
    if k_values is None:
        k_values = _K_VALUES_DIT

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 80)
    LOGGER.info("DiT S3-Cache Stage 0: Normalization + FID Weights")
    LOGGER.info("=" * 80)

    # ------------------------------------------------------------------
    # 步驟 1：載入 Evidence NPZ
    # ------------------------------------------------------------------
    LOGGER.info("\n[Step 1] 載入 Evidence NPZ...")
    l1_int, cosdist_int, svd_int, evidence_meta, B, T = load_dit_evidence(evidence_npz)

    interval_len = T - 1  # = 249 when T=250

    # 產生 block_names = ["block_0", ..., "block_{B-1}"]
    block_names = [f"block_{i}" for i in range(B)]

    # ------------------------------------------------------------------
    # 步驟 2：記錄正規化前統計量
    # ------------------------------------------------------------------
    def _stats(arr: np.ndarray, name: str) -> dict[str, float]:
        valid = arr[np.isfinite(arr)]
        return {
            "name": name,
            "min": float(valid.min()) if len(valid) else float("nan"),
            "max": float(valid.max()) if len(valid) else float("nan"),
            "mean": float(valid.mean()) if len(valid) else float("nan"),
            "std": float(valid.std()) if len(valid) else float("nan"),
            "n_nan": int(np.isnan(arr).sum()),
            "n_inf": int(np.isinf(arr).sum()),
        }

    pre_norm_stats = {
        "l1": _stats(l1_int, "l1_diff"),
        "cosdist": _stats(cosdist_int, "1-cos_sim"),
        "svd": _stats(svd_int, "grassmann_dist"),
    }
    for s in pre_norm_stats.values():
        LOGGER.info(
            f"  Pre-norm {s['name']}: min={s['min']:.6f}, max={s['max']:.6f}, "
            f"mean={s['mean']:.6f}, std={s['std']:.6f}, "
            f"NaN={s['n_nan']}, Inf={s['n_inf']}"
        )

    # ------------------------------------------------------------------
    # 步驟 3：Min-max 正規化
    # ------------------------------------------------------------------
    LOGGER.info("\n[Step 2] Min-max 正規化（三種指標獨立處理）...")
    l1_norm = normalize_minmax(l1_int)
    LOGGER.info(
        f"  l1_interval_norm:     min={l1_norm.min():.4f}, max={l1_norm.max():.4f}, "
        f"mean={l1_norm.mean():.4f}"
    )
    cosdist_norm = normalize_minmax(cosdist_int)
    LOGGER.info(
        f"  cosdist_interval_norm: min={cosdist_norm.min():.4f}, max={cosdist_norm.max():.4f}, "
        f"mean={cosdist_norm.mean():.4f}"
    )
    svd_norm = normalize_minmax(svd_int)
    LOGGER.info(
        f"  svd_interval_norm:     min={svd_norm.min():.4f}, max={svd_norm.max():.4f}, "
        f"mean={svd_norm.mean():.4f}"
    )

    # ------------------------------------------------------------------
    # 步驟 4：載入 FID sensitivity + 計算 block weights
    # ------------------------------------------------------------------
    LOGGER.info("\n[Step 3] 載入 FID sensitivity + 計算 block weights...")
    fid_block_names, delta_fid, baseline_fid = load_dit_delta_fid(fid_json, k_values=k_values)

    # 對齊到 evidence block_names 的順序（DiT 不需要 name mapping，直接對齊）
    missing = [b for b in block_names if b not in delta_fid]
    if missing:
        LOGGER.warning(
            f"以下 {len(missing)} 個 block 在 FID JSON 中找不到 (S_b=0)：{missing}"
        )
    extra = [b for b in fid_block_names if b not in set(block_names)]
    if extra:
        LOGGER.warning(f"FID JSON 中有額外 block（不在 evidence 內，忽略）：{extra}")

    w_clip, w_rank = compute_fid_weights(
        block_names=block_names,
        delta_fid=delta_fid,
        k_values=k_values,
        eps_noise=eps_noise,
        quantile=quantile,
    )

    # ------------------------------------------------------------------
    # 步驟 5：時間軸定義
    # Stage 1 要求: t_curr_interval = np.arange(T-2, -1, -1, dtype=np.int32)
    # 即 [248, 247, ..., 0] 當 T=250
    # ------------------------------------------------------------------
    t_curr_interval = np.arange(T - 2, -1, -1, dtype=np.int32)  # [T-2, T-3, ..., 0]

    axis_def_str = (
        "interval-wise: analysis interval index j (0..T-2) keeps internal order; "
        "display label is t_curr=(T-2)-j"
    )

    # ------------------------------------------------------------------
    # 步驟 6：數值有效性驗證
    # ------------------------------------------------------------------
    LOGGER.info("\n[Step 4] 數值有效性驗證...")
    _check_array(l1_norm, "l1_interval_norm")
    _check_array(cosdist_norm, "cosdist_interval_norm")
    _check_array(svd_norm, "svd_interval_norm")
    _check_array(w_clip, "fid_w_qdiffae_clip")
    _check_array(w_rank, "fid_w_qdiffae_rank")

    # t_curr_interval 驗證
    expected_t_curr = np.arange(T - 2, -1, -1, dtype=np.int32)
    if not np.array_equal(t_curr_interval, expected_t_curr):
        raise RuntimeError(
            f"t_curr_interval 不符合 Stage 1 要求: "
            f"首 4 項 got={t_curr_interval[:4].tolist()}, "
            f"expected={expected_t_curr[:4].tolist()}"
        )
    LOGGER.info(
        f"  ✅ t_curr_interval: [{t_curr_interval[0]}, {t_curr_interval[1]}, ..., {t_curr_interval[-1]}]"
    )

    # ------------------------------------------------------------------
    # 步驟 7：存檔
    # ------------------------------------------------------------------
    LOGGER.info(f"\n[Step 5] 存檔至 {out_path}...")

    block_names_arr = np.array(block_names, dtype=object)
    np.save(out_path / "block_names.npy", block_names_arr)
    np.save(out_path / "l1_interval_norm.npy", l1_norm)
    np.save(out_path / "cosdist_interval_norm.npy", cosdist_norm)
    np.save(out_path / "svd_interval_norm.npy", svd_norm)
    np.save(out_path / "fid_w_qdiffae_clip.npy", w_clip)
    np.save(out_path / "fid_weights.npy", w_clip)           # legacy alias for Stage 1
    np.save(out_path / "fid_w_qdiffae_rank.npy", w_rank)
    np.save(out_path / "t_curr_interval.npy", t_curr_interval)
    np.save(
        out_path / "axis_interval_def.npy",
        np.array(axis_def_str, dtype=object),
    )

    # ------------------------------------------------------------------
    # 步驟 8：Metadata JSON
    # ------------------------------------------------------------------
    l1_denom_mode = evidence_meta.get("l1_denominator", "unknown")
    metadata = {
        "stage": "DiT S3-Cache Stage 0",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": evidence_meta.get("model", "DiT-XL/2"),
        "image_size": evidence_meta.get("image_size", 256),
        "n_blocks": B,
        "n_steps": T,
        "interval_len": interval_len,
        "evidence_npz": str(Path(evidence_npz).resolve()),
        "fid_json": str(Path(fid_json).resolve()),
        "k_values_fid": k_values,
        "eps_noise": eps_noise,
        "quantile": quantile,
        "baseline_fid": baseline_fid,
        "l1_denominator_mode": l1_denom_mode,
        "svd_metric": (
            "grassmann_dist (Grassmann geodesic, NOT Diff-AE's Frobenius-based subspace_dist); "
            "both measure adjacent-timestep subspace drift but differ in formula and matrix decomposed"
        ),
        "l1_source_key": "l1_diff[:, 1:]",
        "cos_source_key": "1 - cos_sim[:, 1:]",
        "svd_source_key": "grassmann_dist[:, 1:]",
        "t_axis_definition": axis_def_str,
        "t_curr_interval_head": t_curr_interval[:4].tolist(),
        "t_curr_interval_tail": t_curr_interval[-4:].tolist(),
        "pre_norm_stats": pre_norm_stats,
        "output_files": [
            "block_names.npy",
            "l1_interval_norm.npy",
            "cosdist_interval_norm.npy",
            "svd_interval_norm.npy",
            "fid_w_qdiffae_clip.npy",
            "fid_weights.npy",
            "fid_w_qdiffae_rank.npy",
            "axis_interval_def.npy",
            "t_curr_interval.npy",
            "stage0_metadata.json",
        ],
    }
    with open(out_path / "stage0_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 結束摘要
    # ------------------------------------------------------------------
    LOGGER.info("\n✅ 全部存檔完成。輸出檔案：")
    for fname in metadata["output_files"]:
        LOGGER.info(f"  {out_path / fname}")

    LOGGER.info("\n[Stage 0 摘要]")
    LOGGER.info(f"  Blocks:       {B}")
    LOGGER.info(f"  T:            {T}  (intervals={interval_len})")
    LOGGER.info(f"  Baseline FID: {baseline_fid:.6f}")
    LOGGER.info(f"  w_clip max:   {w_clip.max():.4f}  (should be 1.0)")
    LOGGER.info(f"  k_values FID: {k_values}")
    LOGGER.info(f"  eps_noise:    {eps_noise}")
    LOGGER.info(f"  quantile:     {quantile}")
    LOGGER.info("=" * 80)
    LOGGER.info("DiT Stage 0 完成！")
    LOGGER.info("=" * 80)


# =============================================================================
# 七、CLI 介面
# =============================================================================

if __name__ == "__main__":
    _repo = Path(__file__).resolve().parents[2]  # DiT/

    parser = argparse.ArgumentParser(
        description="DiT S3-Cache Stage 0: Evidence normalization + FID-based block weights"
    )
    parser.add_argument(
        "--evidence-npz",
        type=str,
        default=str(_repo / "dit_s3cache/outputs/evidence_dit_xl2_256.npz"),
        help="Path to DiT evidence NPZ (default: dit_s3cache/outputs/evidence_dit_xl2_256.npz)",
    )
    parser.add_argument(
        "--fid-json",
        type=str,
        default=str(_repo / "dit_s3cache/fid/fid_sensitivity_results.json"),
        help="Path to FID sensitivity JSON (default: dit_s3cache/fid/fid_sensitivity_results.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_repo / "dit_s3cache/stage0/stage0_output"),
        help="Output directory (default: dit_s3cache/stage0/stage0_output)",
    )
    parser.add_argument(
        "--eps-noise",
        type=float,
        default=0.5,
        help=(
            "Noise correction threshold for FID delta (default: 0.5). "
            "DiT FID@1K numbers may differ in scale from Diff-AE FID@5K; "
            "adjust if all weights collapse to 0."
        ),
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Quantile for FID weight clipping (default: 0.95)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=_K_VALUES_DIT,
        help="FID k values to use (default: 3 5 10)",
    )
    args = parser.parse_args()

    run_stage0_dit(
        evidence_npz=args.evidence_npz,
        fid_json=args.fid_json,
        output_dir=args.output_dir,
        eps_noise=args.eps_noise,
        quantile=args.quantile,
        k_values=args.k_values,
    )
