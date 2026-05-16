"""
Stage-1 baseline（DiT 版）：global shared zones + cost-based per-block k + expanded_mask

時間軸（唯一正式）：
- DDPM 採樣順序 249 → 0：第一步對應單步 timestep t=249，最後一步 t=0。
- expanded_mask[b, i] 的索引 i ∈ [0, T-1] 與「步序」一致：i=0 為 t=249，i=T-1 為 t=0。
  即 ddim_t = (T - 1) - i。

Interval ↔ reused timestep（Stage 0 為 interval-wise evidence）：
- Stage 0 第 j 欄（j=0..T-2）對應 analysis interval j，顯示標籤 t_curr=(T-2)-j。
- 正式語義：interval (t+1 → t) 的不穩定度算在 reused timestep t（reuse 發生在該步）。
- 故欄位 j 對應 reused DDPM timestep t = (T-2) - j（t=0..T-2）。
- t=T-1（此處 t=249）沒有對應 interval 欄位；全域仍強制該步 full compute，
  cutting 信號上 **I_cut[b, T-1] = 0**（不參與 interval 證據）。

相較舊版：已移除 A[b,z]→k 線性映射、zone risk ceiling、舊 regularization；
改為 G[t] 上 smoothing + top-K change points → shared zones，以及 J(b,z,k) 選 k。

命名：I_l1cos = L1/Cos 變化量分支加權（非 stability / similarity）；診斷鍵 `I_l1cos_stats`。

DiT 適配說明：
- B=28 個 DiTBlock，命名 block_0 ~ block_27（flat 架構，無 encoder/mid/decoder 區分）
- T=250 DDPM 步，第一步 t=249，最後一步 t=0
- stage1_name_to_runtime_identity() 直接解析 block_X 格式
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Stage1] %(message)s",
)
LOGGER = logging.getLogger("Stage1Scheduler")

# --- 固定係數（規格）---
# L1/Cos 分支加權（輸入為 Stage 0 正規化「變化量」，非 stability score）
W_L1_BRANCH = 0.7
W_COS_BRANCH = 0.3
ALPHA_ICUT = 4.0 / 9.0
BETA_ICUT = 5.0 / 9.0
assert abs(ALPHA_ICUT + BETA_ICUT - 1.0) < 1e-9

VERSION = "stage1_baseline_v1"
DEFAULT_LAMBDA_SWEEP = (0.25, 0.5, 1.0, 2.0)


def stage1_name_to_runtime_identity(stage1_name: str) -> Tuple[str, int]:
    """DiT block name -> (runtime_name, canonical runtime block id 0..27)."""
    s = stage1_name.strip()
    m = re.match(r"^block_(\d+)$", s)
    if m:
        i = int(m.group(1))
        return f"block_{i}", i
    raise ValueError(f"unrecognized DiT block name: {stage1_name!r}")


def ddim_t_to_step_index(t: int, T: int) -> int:
    """DDPM timestep t → expanded_mask 步序索引 i（i=0 為 t=T-1）。"""
    return (T - 1) - t


def step_index_to_ddim_t(i: int, T: int) -> int:
    """Public function step_index_to_ddim_t."""
    return (T - 1) - i


def interval_j_to_reused_ddim_t(j: int, T: int) -> int:
    """Stage 0 interval 欄 j（0..T-2）→ reused DDPM timestep t（0..T-2）。"""
    return (T - 2) - j


def load_stage0_formal(
    input_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
    """讀入 Stage 0 正式輸出；回傳 block_names, l1, cos, svd, fid_w, axis_def_str, t_curr_interval."""
    p = Path(input_dir)
    required = [
        "block_names.npy",
        "l1_interval_norm.npy",
        "cosdist_interval_norm.npy",
        "svd_interval_norm.npy",
        "fid_w_qdiffae_clip.npy",
        "axis_interval_def.npy",
        "t_curr_interval.npy",
    ]
    for f in required:
        if not (p / f).exists():
            raise FileNotFoundError(f"缺少 Stage 0 正式檔案: {p / f}")

    block_names = np.load(p / "block_names.npy", allow_pickle=True)
    l1 = np.load(p / "l1_interval_norm.npy")
    cos = np.load(p / "cosdist_interval_norm.npy")
    svd = np.load(p / "svd_interval_norm.npy")
    fid_w = np.load(p / "fid_w_qdiffae_clip.npy")
    axis_def = np.load(p / "axis_interval_def.npy", allow_pickle=True)
    t_curr = np.load(p / "t_curr_interval.npy")

    B = len(block_names)
    Tm1 = l1.shape[1] if l1.ndim == 2 else len(l1)
    T = Tm1 + 1
    exp = (B, Tm1)
    for arr, name in [(l1, "l1"), (cos, "cos"), (svd, "svd")]:
        if arr.shape != exp:
            raise ValueError(f"{name} shape {arr.shape} != {exp}")
    if fid_w.shape != (B,):
        raise ValueError(f"fid_w shape {fid_w.shape}")
    if len(t_curr) != Tm1:
        raise ValueError(f"t_curr_interval len {len(t_curr)} != T-1={Tm1}")

    expected_t_curr = np.arange(T - 2, -1, -1, dtype=np.int32)
    t_curr_cmp = np.asarray(t_curr).astype(np.int32).reshape(-1)
    if not np.array_equal(t_curr_cmp, expected_t_curr):
        n_show = min(8, len(t_curr_cmp), len(expected_t_curr))
        raise ValueError(
            "t_curr_interval.npy does not match the expected Stage-0 layout "
            f"np.arange(T-2,-1,-1) for T={T}. "
            "Stage 0 vs Stage 1 interval-to-reused-timestep convention may be inconsistent.\n"
            f"  got (first {n_show}): {t_curr_cmp[:n_show].tolist()}\n"
            f"  expected (first {n_show}): {expected_t_curr[:n_show].tolist()}"
        )

    def clip01(a: np.ndarray, name: str) -> np.ndarray:
        """Public function clip01."""
        if a.min() < 0 or a.max() > 1:
            LOGGER.warning("%s 超出 [0,1]，已 clip", name)
            return np.clip(a, 0, 1)
        return a

    l1 = clip01(l1.astype(np.float64), "l1")
    cos = clip01(cos.astype(np.float64), "cos")
    svd = clip01(svd.astype(np.float64), "svd")
    fid_w = clip01(fid_w.astype(np.float64), "fid_w")

    axis_str = str(axis_def) if np.ndim(axis_def) == 0 else str(axis_def.item())
    if axis_str is None or (
        isinstance(axis_str, str) and (axis_str.strip() == "" or axis_str.strip() == "None")
    ):
        LOGGER.warning(
            "axis_interval_def is empty, None, or missing; cannot verify interval definition from file. "
            "Proceeding, but Stage 0 / Stage 1 convention audits should not rely on this field."
        )

    return block_names, l1, cos, svd, fid_w, axis_str, t_curr_cmp


def build_I_l1cos_I_cut_per_ddim_t(
    l1: np.ndarray,
    cos: np.ndarray,
    svd: np.ndarray,
    T: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    I_l1cos[b,t], I_cut[b,t]，t 為 DDPM timestep 0..T-1。

    I_l1cos is NOT a "similarity" or stability score: it is a combined change / drift
    score from Stage-0 normalized L1 and cosine distance channels (variation magnitudes).
    t=T-1 has no interval column; both arrays are zero there.
    """
    B = l1.shape[0]
    I_l1cos = np.zeros((B, T), dtype=np.float64)
    I_cut = np.zeros((B, T), dtype=np.float64)
    for j in range(T - 1):
        t = interval_j_to_reused_ddim_t(j, T)
        I_l1cos[:, t] = W_L1_BRANCH * l1[:, j] + W_COS_BRANCH * cos[:, j]
        I_cut[:, t] = ALPHA_ICUT * I_l1cos[:, t] + BETA_ICUT * svd[:, j]
    I_l1cos[:, T - 1] = 0.0
    I_cut[:, T - 1] = 0.0
    return I_l1cos, I_cut


def global_cutting_signal_G(I_cut: np.ndarray, fid_w: np.ndarray) -> np.ndarray:
    """
    G[t] = sum_b w_b I_cut[b,t] with normalized weights w_b.

    If fid_w is all ~0, weights fall back to uniform 1/B (explicit; logged).
    Otherwise w is proportional to fid_w (L1-normalized with eps for stability).
    """
    B = I_cut.shape[0]
    eps = 1e-12
    if np.allclose(fid_w, 0.0):
        LOGGER.warning(
            "fid_w_qdiffae_clip is all ~0: using uniform block weights (1/B) for G[t]. "
            "FID-based weighting is disabled for the global cutting signal."
        )
        w = np.full(B, 1.0 / B, dtype=np.float64)
    else:
        w = fid_w.astype(np.float64)
        w = w + eps
        w = w / w.sum()
        LOGGER.debug("G[t]: using FID weights (normalized, eps=%s).", eps)
    return (w[:, None] * I_cut).sum(axis=0)


def validate_shared_zones_ddim(shared_zones: List[Dict[str, Any]], T: int) -> None:
    """
    Fail-fast: partition of DDPM timesteps 0..T-1, ordered zones, no overlap,
    t_start >= t_end, length sum == T.
    """
    covered = np.zeros(T, dtype=bool)
    total_len = 0
    for z in shared_zones:
        zid = z.get("id", "?")
        ts, te = int(z["t_start"]), int(z["t_end"])
        if ts < te:
            raise ValueError(
                f"shared_zones id={zid}: require t_start >= t_end (DDPM order), got t_start={ts}, t_end={te}"
            )
        L = ts - te + 1
        if L < 1:
            raise ValueError(f"shared_zones id={zid}: invalid length L={L}")
        total_len += L
        for t in range(te, ts + 1):
            if not (0 <= t < T):
                raise ValueError(f"shared_zones id={zid}: timestep t={t} out of [0,{T-1}]")
            if covered[t]:
                raise ValueError(
                    f"shared_zones: overlap at DDPM timestep t={t} (zone id={zid}); "
                    "zones must partition 0..T-1 without overlap."
                )
            covered[t] = True
    if total_len != T:
        raise ValueError(
            f"shared_zones: sum of zone lengths is {total_len}, expected T={T} "
            "(full coverage of DDPM timesteps)."
        )
    if not bool(np.all(covered)):
        missing = np.where(~covered)[0].tolist()
        raise ValueError(
            f"shared_zones: missing DDPM timesteps (not covered): first missing indices {missing[:32]}..."
        )


def or_expanded_with_zone_mask(
    expanded_row: np.ndarray,
    ms: np.ndarray,
    block_id: int,
    zone_id: int,
    max_show: int = 32,
) -> None:
    """In-place OR with overlap check (shared_zones should be disjoint in step space)."""
    overlap = expanded_row & ms
    if np.any(overlap):
        idx = np.where(overlap)[0][:max_show]
        raise ValueError(
            "expanded_mask overlap when merging zones: "
            f"block_id={block_id}, zone_id={zone_id}, "
            f"overlapping step indices (first {len(idx)}): {idx.tolist()}"
        )
    expanded_row |= ms


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Public function moving_average."""
    if window <= 1:
        return x.copy()
    k = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, k, mode="same")


def processing_order_series(G_ddim: np.ndarray, T: int) -> np.ndarray:
    """G_proc[i] = G_ddim[t=(T-1)-i]，長度 T；i=0 為第一步 t=T-1。"""
    g = np.zeros(T, dtype=np.float64)
    for i in range(T):
        t = step_index_to_ddim_t(i, T)
        g[i] = G_ddim[t]
    return g


def delta_adjacent(g: np.ndarray) -> np.ndarray:
    """g 長度 T；Delta[i]=|g[i]-g[i-1]|，i=1..T-1；回傳長度 T（Delta[0]=0）。"""
    T = len(g)
    d = np.zeros(T, dtype=np.float64)
    for i in range(1, T):
        d[i] = abs(g[i] - g[i - 1])
    return d


def topk_change_point_indices(
    Delta: np.ndarray,
    K: int,
    candidate_lo: int,
    candidate_hi: int,
) -> List[int]:
    """
    在 [candidate_lo, candidate_hi] 內取 Delta 最大的 K 個 邊界索引 i
    （表示新 zone 從 step index i 開始）。tie-break：索引較大者優先（與 argsort 穩定性配合）。
    """
    cand = list(range(candidate_lo, candidate_hi + 1))
    if K <= 0:
        return []
    if not cand:
        return []
    vals = np.array([Delta[i] for i in cand], dtype=np.float64)
    # 由大到小；tie 時較大索引先（np.lexsort）
    order = np.lexsort((-np.arange(len(cand)), -vals))
    topk = min(K, len(cand))
    picked = sorted([cand[order[j]] for j in range(topk)])
    return picked


def zones_from_step_boundaries(boundaries: List[int]) -> List[Tuple[int, int]]:
    """boundaries 遞增、首 0 末 T；回傳 (s_start, s_end) step index 閉區間。"""
    zones = []
    for z in range(len(boundaries) - 1):
        s0, s1e = boundaries[z], boundaries[z + 1] - 1
        if s0 <= s1e:
            zones.append((s0, s1e))
    return zones


def step_zone_to_ddim_zone(s0: int, s1: int, T: int) -> Tuple[int, int]:
    """
    step index zone [s0,s1]（含）→ DDPM zone [t_start,t_end]（含），
    處理順序先 t_start 後 t_end，且 t_start >= t_end。
    """
    t_start = step_index_to_ddim_t(s0, T)
    t_end = step_index_to_ddim_t(s1, T)
    return int(t_start), int(t_end)


def merge_short_zones_step(
    zones: List[Tuple[int, int]],
    T: int,
    min_len: int = 2,
) -> List[Tuple[int, int]]:
    """
    zones 為 step index 閉區間、有序且完整覆蓋 [0,T-1]。
    長度 < min_len：預設 merge right；若已是最後一個 zone 則 merge left。
    重複直到無短 zone 或無法再併（單一 zone）。
    """
    zlist = [list(z) for z in zones]
    changed = True
    guard = 0
    while changed and guard < T * 4:
        guard += 1
        changed = False
        i = 0
        while i < len(zlist):
            s0, s1 = zlist[i][0], zlist[i][1]
            L = s1 - s0 + 1
            if L >= min_len:
                i += 1
                continue
            if len(zlist) == 1:
                break
            if i < len(zlist) - 1:
                zlist[i][1] = zlist[i + 1][1]
                zlist.pop(i + 1)
                changed = True
            else:
                zlist[i - 1][1] = zlist[i][1]
                zlist.pop(i)
                changed = True
    return [(int(a[0]), int(a[1])) for a in zlist]


def zone_fr_pattern(L: int, k: int) -> Tuple[bool, ...]:
    """長度 L：位置 0,k,2k,... 為 F（zone 內相對位置）。"""
    return tuple((p % k == 0) for p in range(L))


def unique_k_representatives(L: int, k_min: int, k_max: int) -> List[int]:
    """同 pattern 只保留最小 k 作 representative。"""
    seen: Dict[Tuple[bool, ...], int] = {}
    for k in range(k_min, k_max + 1):
        pat = zone_fr_pattern(L, k)
        if pat not in seen:
            seen[pat] = k
    return sorted(seen.values())


def expand_zone_mask_ddim(
    t_start: int,
    t_end: int,
    k: int,
    T: int,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    回傳：
    - mask_step: (T,) bool，步序 i 上是否 F
    - F_timesteps, R_timesteps（DDPM t 列表）
    """
    L = t_start - t_end + 1
    mask_step = np.zeros(T, dtype=bool)
    F_ts: List[int] = []
    R_ts: List[int] = []
    for p in range(L):
        t = t_start - p
        is_f = p % k == 0
        i = ddim_t_to_step_index(t, T)
        mask_step[i] = is_f
        if is_f:
            F_ts.append(t)
        else:
            R_ts.append(t)
    return mask_step, F_ts, R_ts


def cost_J_for_k(
    I_cut_b: np.ndarray,
    fid_w_b: float,
    t_start: int,
    t_end: int,
    k: int,
    L: int,
    lam: float,
    T: int,
) -> Tuple[float, float, float, int, int]:
    """
    J = w * (1/L) * sum_{t in R} I_cut[t] + lam * |F|/L
    回傳 (J, reuse_term_unweighted_sum/L, penalty, n_R, n_F)
    """
    _, F_ts, R_ts = expand_zone_mask_ddim(t_start, t_end, k, T)
    n_F = len(F_ts)
    n_R = len(R_ts)
    reuse_sum = sum(I_cut_b[t] for t in R_ts) if R_ts else 0.0
    r_term = reuse_sum / L
    pen = lam * (n_F / L)
    J = float(fid_w_b) * r_term + pen
    return J, float(r_term), float(pen), n_R, n_F


def stats_dict(x: np.ndarray) -> Dict[str, float]:
    """Public function stats_dict."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def run_stage1_synthesis(
    stage0_dir: str,
    output_dir: str,
    K: int = 10,
    smooth_window: int = 5,
    lambda_base: float = 1.0,
    lambda_sweep: Sequence[float] = DEFAULT_LAMBDA_SWEEP,
    k_min: int = 1,
    k_max: int = 4,
    min_zone_len: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Public function run_stage1_synthesis."""
    block_names, l1, cos, svd, fid_w, axis_def_str, t_curr_arr = load_stage0_formal(stage0_dir)
    B = l1.shape[0]
    T = l1.shape[1] + 1

    I_l1cos, I_cut = build_I_l1cos_I_cut_per_ddim_t(l1, cos, svd, T)
    G_ddim = global_cutting_signal_G(I_cut, fid_w)
    G_proc = processing_order_series(G_ddim, T)
    G_smooth = moving_average(G_proc, smooth_window)
    Delta = delta_adjacent(G_smooth)

    # 候選邊界為步序 i=1..T-1（新 zone 從 i 開始）；最多取 T-1 個
    K_eff = min(int(K), max(0, T - 1))
    cps = topk_change_point_indices(Delta, K_eff, 1, T - 1)
    boundaries = sorted(set([0] + cps + [T]))
    step_zones = zones_from_step_boundaries(boundaries)
    step_zones = merge_short_zones_step(step_zones, T, min_len=min_zone_len)

    shared_zones: List[Dict[str, Any]] = []
    for zi, (s0, s1) in enumerate(step_zones):
        ts, te = step_zone_to_ddim_zone(s0, s1, T)
        shared_zones.append(
            {
                "id": zi,
                "t_start": ts,
                "t_end": te,
                "length": ts - te + 1,
            }
        )

    validate_shared_zones_ddim(shared_zones, T)

    Z = len(shared_zones)

    # --- 每 (b,z) 候選 k 與 cost ---
    cand_k_per_zone: List[List[int]] = []
    cost_tables: List[Dict[str, Any]] = []
    k_chosen = np.zeros((B, Z), dtype=np.int32)

    for zi, zd in enumerate(shared_zones):
        Lz = zd["length"]
        ts, te = zd["t_start"], zd["t_end"]
        cands = unique_k_representatives(Lz, k_min, k_max)
        cand_k_per_zone.append(cands)

        zone_cost: Dict[str, Any] = {
            "zone_id": zi,
            "length": Lz,
            "candidate_k": cands,
            "per_block": [],
        }

        for b in range(B):
            row: Dict[str, Any] = {"block_id": b, "candidates": {}}
            runtime_name, runtime_block_id = stage1_name_to_runtime_identity(str(block_names[b]))
            best_k = cands[0]
            best_J = float("inf")
            for k in cands:
                J, rterm, pen, nR, nF = cost_J_for_k(
                    I_cut[b], fid_w[b], ts, te, k, Lz, lambda_base, T
                )
                row["candidates"][str(k)] = {
                    "J": J,
                    "reuse_risk_term_would_be_unweighted": rterm,
                    "fid_w_on_reuse_term": float(fid_w[b]),
                    "compute_penalty_lambda_f_over_L": pen,
                    "n_R": nR,
                    "n_F": nF,
                }
                if J < best_J - 1e-15:
                    best_J = J
                    best_k = k
            row["selected_k"] = int(best_k)
            row["selected_J"] = float(best_J)
            row["scheduler_local_block_id"] = int(b)
            row["runtime_name"] = runtime_name
            row["canonical_runtime_block_id"] = int(runtime_block_id)
            k_chosen[b, zi] = int(best_k)
            zone_cost["per_block"].append(row)

        cost_tables.append(zone_cost)

    # --- expanded_mask [B, T]，步序 i：i=0 -> t=T-1 ---
    expanded = np.zeros((B, T), dtype=bool)
    for b in range(B):
        for zi, zd in enumerate(shared_zones):
            ms, _, _ = expand_zone_mask_ddim(zd["t_start"], zd["t_end"], int(k_chosen[b, zi]), T)
            or_expanded_with_zone_mask(expanded[b, :], ms, block_id=b, zone_id=int(zd["id"]))
    # 強制 t=T-1（i=0）為 F
    expanded[:, 0] = True

    blocks_out: List[Dict[str, Any]] = []
    total_cost_b = np.zeros(B, dtype=np.float64)
    for b in range(B):
        tc = 0.0
        block_name = str(block_names[b])
        runtime_name, runtime_block_id = stage1_name_to_runtime_identity(block_name)
        for zi, zd in enumerate(shared_zones):
            Lz = zd["length"]
            J, _, _, _, _ = cost_J_for_k(
                I_cut[b],
                fid_w[b],
                zd["t_start"],
                zd["t_end"],
                int(k_chosen[b, zi]),
                Lz,
                lambda_base,
                T,
            )
            tc += J
        total_cost_b[b] = tc
        blocks_out.append(
            {
                "id": b,
                "scheduler_local_block_id": b,
                "block_id_semantics": "scheduler-local index (not canonical runtime block index)",
                "name": block_name,
                "runtime_name": runtime_name,
                "canonical_runtime_block_id": int(runtime_block_id),
                "k_per_zone": k_chosen[b, :].tolist(),
                "expanded_mask": expanded[b, :].tolist(),
            }
        )

    nF_per = expanded.sum(axis=1)
    nR_per = T - nF_per

    stage1_baseline_params = {
        "K_change_points": int(K),
        "K_effective_used": int(K_eff),
        "smooth_window": int(smooth_window),
        "lambda": float(lambda_base),
        "lambda_sweep": [float(x) for x in lambda_sweep],
        "k_min": int(k_min),
        "k_max": int(k_max),
        "min_zone_len": int(min_zone_len),
        "W_L1_branch": W_L1_BRANCH,
        "W_COS_branch": W_COS_BRANCH,
        "ALPHA_ICUT": ALPHA_ICUT,
        "BETA_ICUT": BETA_ICUT,
        "I_cut_formula": (
            "I_cut = (4/9)*I_l1cos + (5/9)*SVD; "
            "I_l1cos = 0.7*L1_norm + 0.3*Cos_norm (change magnitudes, not stability); "
            "interval columns mapped to reused DDPM t"
        ),
    }

    config: Dict[str, Any] = {
        "version": VERSION,
        "T": T,
        "time_order": "ddpm_249_to_0",
        "expanded_mask_layout": "index i=0 is first DDPM step (t=249); i=249 is t=0",
        "block_identity_semantics": {
            "blocks[].id": "scheduler-local index (stable within this scheduler JSON)",
            "blocks[].scheduler_local_block_id": "same as blocks[].id",
            "blocks[].canonical_runtime_block_id": "canonical runtime index aligned with runtime layer order",
        },
        "stage1_baseline_params": stage1_baseline_params,
        "shared_zones": shared_zones,
        "blocks": blocks_out,
    }

    # --- lambda sweep：用分解項快速算 alternative selected k ---
    sweep_report: Dict[str, Any] = {}
    for lam in lambda_sweep:
        key = str(lam)
        sweep_report[key] = {"per_block_k": []}
        for b in range(B):
            k_z = []
            for zi, zd in enumerate(shared_zones):
                Lz = zd["length"]
                ts, te = zd["t_start"], zd["t_end"]
                best_k = cand_k_per_zone[zi][0]
                best_J = float("inf")
                for k in cand_k_per_zone[zi]:
                    J, _, _, _, _ = cost_J_for_k(I_cut[b], fid_w[b], ts, te, k, Lz, float(lam), T)
                    if J < best_J - 1e-15:
                        best_J = J
                        best_k = k
                k_z.append(int(best_k))
            sweep_report[key]["per_block_k"].append(k_z)

    diagnostics: Dict[str, Any] = {
        "stage0_axis_interval_def": axis_def_str,
        "t_curr_interval": t_curr_arr.tolist(),
        "mapping_note": "interval j -> reused DDPM t = (T-2)-j; I_cut[:,T-1]=0; expanded_mask[i] step maps to t=(T-1)-i",
        "I_l1cos_stats": stats_dict(I_l1cos),
        "I_cut_stats": stats_dict(I_cut),
        "G_ddim": G_ddim.tolist(),
        "G_processing_order_i0_is_t249": G_proc.tolist(),
        "G_smooth_processing_order": G_smooth.tolist(),
        "Delta_processing_order": Delta.tolist(),
        "change_points_step_index": cps,
        "step_zones_before_merge": [
            {"s0": a, "s1": b} for a, b in zones_from_step_boundaries(boundaries)
        ],
        "merged_step_zones": [{"s0": a, "s1": b} for a, b in step_zones],
        "shared_zones_ddim": shared_zones,
        "candidate_k_per_zone": cand_k_per_zone,
        "cost_tables_per_zone": cost_tables,
        "baseline_lambda": float(lambda_base),
        "lambda_sweep": [float(x) for x in lambda_sweep],
        "lambda_sweep_selected_k": sweep_report,
    }

    # verification_summary
    zone_boundary_list = [{"t_start": z["t_start"], "t_end": z["t_end"]} for z in shared_zones]
    per_zone_ver: List[Dict[str, Any]] = []
    for zi, zd in enumerate(shared_zones):
        per_block_J = [float(cost_tables[zi]["per_block"][b]["selected_J"]) for b in range(B)]
        per_zone_ver.append(
            {
                "zone_id": zi,
                "length": zd["length"],
                "candidate_k": cand_k_per_zone[zi],
                "selected_k_per_block": k_chosen[:, zi].tolist(),
                "selected_J_per_block": per_block_J,
                "cost_table_ref": f"scheduler_diagnostics.json -> cost_tables_per_zone[{zi}]",
            }
        )

    verification_summary: Dict[str, Any] = {
        "shared_zone_boundaries": zone_boundary_list,
        "per_block": [
            {
                "block_id": b,
                "k_per_zone": k_chosen[b, :].tolist(),
                "num_F": int(nF_per[b]),
                "num_R": int(nR_per[b]),
                "total_cost_J_sum_zones": float(total_cost_b[b]),
            }
            for b in range(B)
        ],
        "per_zone": per_zone_ver,
        "lambda_baseline": float(lambda_base),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scheduler_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(out / "scheduler_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    with open(out / "verification_summary.json", "w", encoding="utf-8") as f:
        json.dump(verification_summary, f, indent=2, ensure_ascii=False)

    LOGGER.info("已寫入 %s", out)
    return config, diagnostics, verification_summary


def rebuild_expanded_mask_from_config(config: Dict[str, Any]) -> np.ndarray:
    """由 shared_zones + k_per_zone 重建 [B,T] expanded_mask（步序與本模組一致）。"""
    T = int(config["T"])
    zones = config["shared_zones"]
    B = len(config["blocks"])
    m = np.zeros((B, T), dtype=bool)
    for b, block in enumerate(config["blocks"]):
        bid = int(block.get("id", b))
        for z, k in zip(zones, block["k_per_zone"]):
            ms, _, _ = expand_zone_mask_ddim(z["t_start"], z["t_end"], int(k), T)
            or_expanded_with_zone_mask(m[b, :], ms, block_id=bid, zone_id=int(z["id"]))
        m[b, 0] = True
    return m


def _self_test_fid_zero_uniform_G() -> None:
    """fid_w 全 0 時 G 為 block 平均（uniform weights）。"""
    B, T = 4, 250
    I_cut = np.arange(B * T, dtype=np.float64).reshape(B, T)
    fid = np.zeros(B, dtype=np.float64)
    G = global_cutting_signal_G(I_cut, fid)
    exp = I_cut.mean(axis=0)
    assert np.allclose(G, exp), (G[:5], exp[:5])


def _self_test_t_curr_mismatch_raises(tmp: Path, B: int, T: int) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    Tm1 = T - 1
    test_names = np.array(
        [f"block_{i}" for i in range(B)],
        dtype=object,
    )
    np.save(tmp / "block_names.npy", test_names)
    np.save(tmp / "l1_interval_norm.npy", np.zeros((B, Tm1)))
    np.save(tmp / "cosdist_interval_norm.npy", np.zeros((B, Tm1)))
    np.save(tmp / "svd_interval_norm.npy", np.zeros((B, Tm1)))
    np.save(tmp / "fid_w_qdiffae_clip.npy", np.ones(B))
    np.save(tmp / "t_curr_interval.npy", np.zeros(Tm1, dtype=np.int32))  # wrong layout
    np.save(tmp / "axis_interval_def.npy", np.array("ok", dtype=object))
    try:
        load_stage0_formal(str(tmp))
    except ValueError as e:
        assert "t_curr_interval" in str(e) or "expected" in str(e).lower()
        return
    raise AssertionError("expected ValueError for bad t_curr_interval")


def _self_test_merge_zones_cover_T() -> None:
    """merge_short_zones_step 後仍完整覆蓋 step 0..T-1。"""
    T = 250
    raw = [(i, i) for i in range(T)]  # T singletons
    merged = merge_short_zones_step(raw, T, min_len=2)
    covered = np.zeros(T, dtype=bool)
    for s0, s1 in merged:
        covered[s0 : s1 + 1] = True
    assert covered.all() and len(merged) >= 1


def self_test() -> Any:
    """Public function self_test."""
    T = 250
    B = 3
    Tm1 = T - 1
    tmp = Path("/tmp/stage1_dit_self_test")
    tmp.mkdir(parents=True, exist_ok=True)

    _self_test_fid_zero_uniform_G()
    _self_test_t_curr_mismatch_raises(tmp / "bad_tcurr", B, T)
    _self_test_merge_zones_cover_T()

    test_names = np.array(
        [f"block_{i}" for i in range(B)],
        dtype=object,
    )
    np.save(tmp / "block_names.npy", test_names)
    rng = np.random.default_rng(0)
    np.save(tmp / "l1_interval_norm.npy", rng.random((B, Tm1)).astype(np.float64))
    np.save(tmp / "cosdist_interval_norm.npy", rng.random((B, Tm1)).astype(np.float64))
    np.save(tmp / "svd_interval_norm.npy", rng.random((B, Tm1)).astype(np.float64))
    np.save(tmp / "fid_w_qdiffae_clip.npy", rng.random(B).astype(np.float64))
    np.save(tmp / "t_curr_interval.npy", (Tm1 - 1) - np.arange(Tm1, dtype=np.int32))
    np.save(
        tmp / "axis_interval_def.npy",
        np.array("test", dtype=object),
    )
    outd = tmp / "out"
    cfg, _, _ = run_stage1_synthesis(
        str(tmp),
        str(outd),
        K=5,
        smooth_window=3,
        lambda_base=1.0,
        k_max=4,
    )
    validate_shared_zones_ddim(cfg["shared_zones"], T)
    recon = rebuild_expanded_mask_from_config(cfg)
    stacked = np.array([cfg["blocks"][b]["expanded_mask"] for b in range(B)], dtype=bool)
    assert np.array_equal(recon, stacked)
    assert cfg["blocks"][0]["expanded_mask"][0] is True
    assert cfg["time_order"] == "ddpm_249_to_0", f"unexpected time_order: {cfg['time_order']}"
    assert cfg["T"] == T, f"unexpected T: {cfg['T']}"
    print("self_test OK")


def main() -> Any:
    """Public function main."""
    import argparse

    p = argparse.ArgumentParser(description="Stage-1 baseline scheduler (DiT)")
    p.add_argument("--stage0_dir", type=str, default="dit_s3cache/stage0/stage0_output")
    p.add_argument("--output_dir", type=str, default="dit_s3cache/stage1/stage1_output")
    p.add_argument("--K", type=int, default=10, help="top-K change points（內部會 cap）")
    p.add_argument("--smooth_window", type=int, default=5)
    p.add_argument("--lambda", dest="lambda_base", type=float, default=1.0)
    p.add_argument(
        "--lambda_sweep",
        type=str,
        default="0.25,0.5,1.0,2.0",
        help="逗號分隔，寫入 diagnostics",
    )
    p.add_argument("--k_min", type=int, default=1)
    p.add_argument("--k_max", type=int, default=4)
    p.add_argument("--min_zone_len", type=int, default=2)
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    sweep = tuple(float(x.strip()) for x in args.lambda_sweep.split(",") if x.strip())
    run_stage1_synthesis(
        args.stage0_dir,
        args.output_dir,
        K=args.K,
        smooth_window=args.smooth_window,
        lambda_base=args.lambda_base,
        lambda_sweep=sweep,
        k_min=args.k_min,
        k_max=args.k_max,
        min_zone_len=args.min_zone_len,
    )
    print("Done:", args.output_dir)


if __name__ == "__main__":
    main()
