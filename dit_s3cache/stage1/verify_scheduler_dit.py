"""
驗證 DiT Stage-1：shared_zones + k_per_zone + expanded_mask（DDPM 步序 249→0）。

步序 i：i=0 為 t=249（第一步），i=T-1 為 t=0。
expanded_mask[b,i]==True 為 full compute (F)，False 為 reuse (R)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np


def load_config(path: str) -> Dict[str, Any]:
    """Public function load_config."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_zone_mask_ddim(t_start: int, t_end: int, k: int, T: int) -> np.ndarray:
    """與 stage1_scheduler_dit 一致：回傳 (T,) bool，索引為步序 i（i=0 -> t=T-1）。"""
    L = t_start - t_end + 1
    mask_step = np.zeros(T, dtype=bool)
    for p in range(L):
        t = t_start - p
        is_f = p % k == 0
        i = (T - 1) - t
        mask_step[i] = is_f
    return mask_step


def rebuild_mask(shared_zones: List[Dict[str, Any]], k_per_zone: List[int], T: int) -> np.ndarray:
    """Public function rebuild_mask."""
    m = np.zeros(T, dtype=bool)
    for z, k in zip(shared_zones, k_per_zone):
        m |= expand_zone_mask_ddim(int(z["t_start"]), int(z["t_end"]), int(k), T)
    m[0] = True
    return m


def check_shared_zones_cover_ddim(shared_zones: List[Dict[str, Any]], T: int) -> bool:
    """每個 DDPM timestep t ∈ [0,T-1] 恰落在一個 zone [t_start,t_end]（t_start >= t_end）。"""
    covered = np.zeros(T, dtype=bool)
    for z in shared_zones:
        ts, te = int(z["t_start"]), int(z["t_end"])
        if ts < te:
            print(f"❌ zone id={z.get('id')} 需 t_start >= t_end，收到 t_start={ts}, t_end={te}")
            return False
        for t in range(te, ts + 1):
            if covered[t]:
                print(f"❌ DDPM t={t} 被多個 zone 覆蓋")
                return False
            covered[t] = True
    if not covered.all():
        print(f"❌ 未覆蓋的 DDPM t: {np.where(~covered)[0].tolist()}")
        return False
    print("✅ shared_zones 完整、不重疊覆蓋所有 DDPM timestep")
    return True


def check_time_order(cfg: Dict[str, Any]) -> bool:
    """Public function check_time_order."""
    order = cfg.get("time_order")
    if order != "ddpm_249_to_0":
        print(f"❌ time_order 必須為 'ddpm_249_to_0'，收到: {order!r}")
        return False
    print("✅ time_order 正確（ddpm_249_to_0）")
    return True


def check_block_ids(blocks: List[Dict[str, Any]]) -> bool:
    """Public function check_block_ids."""
    B = len(blocks)
    seen: Set[int] = set()
    ok = True
    for idx, block in enumerate(blocks):
        if "id" not in block:
            print(f"❌ block index={idx} 缺少 id")
            ok = False
            continue
        bid = block["id"]
        if not isinstance(bid, int):
            print(f"❌ block index={idx} id 必須為 int，收到 {type(bid).__name__}")
            ok = False
            continue
        if bid < 0 or bid >= B:
            print(f"❌ block id={bid} 超出合法範圍 [0, {B - 1}]")
            ok = False
        if bid in seen:
            print(f"❌ block id 重複: {bid}")
            ok = False
        seen.add(bid)
    if len(seen) != B:
        miss = sorted(set(range(B)) - seen)
        if miss:
            print(f"❌ block id 非連續/缺漏，缺少: {miss}")
        ok = False
    if ok:
        print("✅ block 數量與 id 唯一性/範圍檢查通過")
    return ok


def check_shared_zone_ids(shared_zones: List[Dict[str, Any]]) -> bool:
    """Public function check_shared_zone_ids."""
    Z = len(shared_zones)
    seen: Set[int] = set()
    ok = True
    for idx, z in enumerate(shared_zones):
        zid = z.get("id", None)
        if not isinstance(zid, int):
            print(f"❌ shared_zones[{idx}] id 必須為 int，收到 {zid!r}")
            ok = False
            continue
        if zid < 0 or zid >= Z:
            print(f"❌ zone id={zid} 超出合法範圍 [0, {Z - 1}]")
            ok = False
        if zid in seen:
            print(f"❌ zone id 重複: {zid}")
            ok = False
        seen.add(zid)
        if "length" in z:
            ts = int(z["t_start"])
            te = int(z["t_end"])
            exp_len = ts - te + 1
            if int(z["length"]) != exp_len:
                print(f"❌ zone id={zid} length={z['length']} 與 t_start/t_end 不一致（應為 {exp_len}）")
                ok = False
    if len(seen) != Z:
        miss = sorted(set(range(Z)) - seen)
        if miss:
            print(f"❌ zone id 非連續/缺漏，缺少: {miss}")
        ok = False
    if ok:
        print("✅ shared_zones 的 id 與 length 一致性檢查通過")
    return ok


def main() -> int:
    """Public function main."""
    parser = argparse.ArgumentParser(description="Verify Stage-1 baseline scheduler JSON (DiT)")
    parser.add_argument(
        "--config",
        type=str,
        default="dit_s3cache/stage1/stage1_output/scheduler_config.json",
    )
    args = parser.parse_args()

    path = Path(args.config)
    if not path.exists():
        print(f"❌ 找不到 {path}")
        return 1

    cfg = load_config(str(path))
    T = int(cfg["T"])
    shared_zones = cfg["shared_zones"]
    blocks = cfg["blocks"]
    params = cfg.get("stage1_baseline_params", {})
    k_min = int(params.get("k_min", 1))
    k_max = int(params.get("k_max", 4))

    print("=" * 72)
    print("Stage-1 baseline 驗證（DiT）")
    print("=" * 72)
    print(f"time_order: {cfg.get('time_order')}")
    print(f"T={T}, |shared_zones|={len(shared_zones)}, blocks={len(blocks)}")

    all_ok = True
    all_ok &= check_time_order(cfg)
    all_ok &= check_block_ids(blocks)
    all_ok &= check_shared_zone_ids(shared_zones)
    all_ok &= check_shared_zones_cover_ddim(shared_zones, T)

    exp_all_raw = [b.get("expanded_mask") for b in blocks]
    if any(not isinstance(m, list) for m in exp_all_raw):
        print("❌ 每個 block 都必須有 list 型別的 expanded_mask")
        return 1
    exp_all = np.array(exp_all_raw, dtype=bool)
    if exp_all.ndim != 2:
        print(f"❌ expanded_mask 應為 [blocks, T] 二維結構，收到 ndim={exp_all.ndim}")
        return 1
    if exp_all.shape[1] != T:
        print(f"❌ expanded_mask 寬度 {exp_all.shape[1]} != T={T}")
        all_ok = False

    if not exp_all[:, 0].all():
        print("❌ 所有 block 在步序 i=0（DDPM t=249）必須為 F（True）")
        all_ok = False
    else:
        print("✅ 所有 block：步序 i=0（t=249）為 F")

    zone_start_ok = True
    for z in shared_zones:
        ts = int(z["t_start"])
        i0 = (T - 1) - ts
        if not exp_all[:, i0].all():
            print(f"❌ zone id={z['id']} 起點 t={ts}（步序 i={i0}）須全為 F")
            zone_start_ok = False
            all_ok = False
    if zone_start_ok:
        print("✅ 每個 zone 在該區第一步（DDPM t=t_start）為 F")

    for block in blocks:
        bid = block["id"]
        kz = block["k_per_zone"]
        if len(kz) != len(shared_zones):
            print(f"❌ block {bid} k_per_zone 長度 {len(kz)} != {len(shared_zones)}")
            all_ok = False
            continue
        arr = np.array(kz, dtype=int)
        if (arr < k_min).any() or (arr > k_max).any():
            print(f"❌ block {bid} k 超出 [{k_min}, {k_max}]: {kz}")
            all_ok = False

    if all_ok:
        print(f"✅ 所有 k 在 [{k_min}, {k_max}]")

    for b, block in enumerate(blocks):
        recon = rebuild_mask(shared_zones, block["k_per_zone"], T)
        if not np.array_equal(exp_all[b], recon):
            diff = np.where(exp_all[b] != recon)[0]
            print(f"❌ block {b} expanded_mask 與重建不符，相異步序（前 30 個）: {diff[:30].tolist()}")
            all_ok = False
    if all_ok:
        print("✅ 每個 block 的 expanded_mask 與 shared_zones + k_per_zone 重建一致")

    nF = exp_all.sum(axis=1)
    nR = T - nF
    print("\n📊 每 block #F / #R:")
    for b in range(len(blocks)):
        print(f"   block {b} ({blocks[b].get('name', '')}): F={int(nF[b])}, R={int(nR[b])}")

    print("\n" + ("🎉 全部通過" if all_ok else "⚠️ 有項目失敗"))
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
