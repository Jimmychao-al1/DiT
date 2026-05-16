"""
Stage2 (DiT): 將 Stage1 的 scheduler_config.json 轉成 runtime 可用的 cache scheduler。

時間軸（與 Stage1-DiT 一致）：
- DDPM 採樣步序 index i=0..T-1，其中 i=0 對應 DDPM t=T-1=249（第一步），i=T-1 對應 t=0（最後一步）。
- Stage1 的 expanded_mask 使用同一個步序 index。
- runtime cache scheduler 的 value 是「需要 full compute 的 step index 集合」。
- step index 是診斷 JSON 的 key（per user choice: (a) step_index）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dit_s3cache.stage1.stage1_scheduler_dit import (
    expand_zone_mask_ddim,
    or_expanded_with_zone_mask,
    validate_shared_zones_ddim,
)

# --- DiT-specific constants ---
DIT_T = 250                             # 總步數
DIT_N_INTERVALS = 249                   # T - 1
DIT_NUM_BLOCKS = 28                     # block_0 ~ block_27
TIME_ORDER_EXPECTED = "ddpm_249_to_0"   # 對應 Stage1 JSON 的 time_order

EXPECTED_NUM_BLOCKS = DIT_NUM_BLOCKS

# DiT 是 flat 架構：runtime 名稱即為 Stage1 canonical 名稱（block_0 ~ block_27）
RUNTIME_LAYER_NAMES: Tuple[str, ...] = tuple(f"block_{i}" for i in range(DIT_NUM_BLOCKS))
assert len(RUNTIME_LAYER_NAMES) == EXPECTED_NUM_BLOCKS

# Stage 2 error collection 從第一步必須 full compute（step_idx=0 → DDPM t=249）
FIRST_STEP_IDX = 0


def load_stage1_scheduler_config(path: str | Path) -> Dict[str, Any]:
    """讀取 Stage1 scheduler_config.json。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"scheduler_config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError("scheduler_config root must be a JSON object")
    return cfg


def runtime_name_to_block_id(runtime: str) -> int:
    """Runtime 名稱 'block_X' → canonical runtime block id（0~27）。"""
    s = runtime.strip()
    m = re.match(r"^block_(\d+)$", s)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < DIT_NUM_BLOCKS:
            return idx
    raise ValueError(f"unrecognized runtime block name: {runtime!r}")


def stage1_block_to_runtime_block(stage1_name: str) -> str:
    """Stage1 canonical 名稱 → Runtime 名稱（DiT flat 架構，恆等映射）。"""
    s = stage1_name.strip()
    if re.match(r"^block_\d+$", s):
        return s
    raise ValueError(f"unrecognized Stage1 block name: {stage1_name!r}")


def runtime_block_to_stage1_name(runtime: str) -> str:
    """Runtime 名稱 → Stage1 canonical 名稱（DiT flat 架構，恆等映射）。"""
    s = runtime.strip()
    if re.match(r"^block_\d+$", s):
        return s
    raise ValueError(f"unrecognized runtime block name: {runtime!r}")


def step_index_to_ddpm_t(step_idx: int, T: int) -> int:
    """步序索引 → DDPM timestep t（i=0 → t=T-1）。"""
    return (T - 1) - int(step_idx)


def ddpm_t_to_step_index(t: int, T: int) -> int:
    """DDPM timestep t → 步序索引（t=T-1 → i=0）。"""
    return (T - 1) - int(t)


def rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
    shared_zones: List[Dict[str, Any]],
    k_per_zone: List[int],
    T: int,
    *,
    block_id: int = 0,
) -> List[bool]:
    """
    由 shared_zones + k_per_zone 重建單一 block 的 expanded_mask（長度 T）。
    規則對齊 Stage1：zone union 後強制 expanded_mask[0] = True（第一步必須 full compute）。
    """
    if len(k_per_zone) != len(shared_zones):
        raise ValueError(
            f"k_per_zone len {len(k_per_zone)} != shared_zones len {len(shared_zones)}"
        )
    row = np.zeros(T, dtype=bool)
    for z, k in zip(shared_zones, k_per_zone):
        ms, _, _ = expand_zone_mask_ddim(int(z["t_start"]), int(z["t_end"]), int(k), T)
        or_expanded_with_zone_mask(row, ms, block_id=block_id, zone_id=int(z["id"]))
    row[0] = True  # step_idx=0 (DDPM t=T-1) 必須 full compute
    return row.tolist()


def validate_stage1_scheduler_config(
    cfg: Dict[str, Any],
    *,
    require_full_coverage: bool = True,
    require_k_per_zone: bool = True,
) -> None:
    """
    Stage1/Stage2 scheduler 結構驗證。

    require_k_per_zone=False 時，只檢查 expanded_mask 合法，不驗證 rebuild(k) 一致性。
    """
    if cfg.get("time_order") != TIME_ORDER_EXPECTED:
        raise ValueError(
            f"time_order must be {TIME_ORDER_EXPECTED!r}, got {cfg.get('time_order')!r}"
        )

    T = int(cfg["T"])
    if T < 2:
        raise ValueError(f"T must be >= 2, got {T}")

    blocks = cfg.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a list")
    if require_full_coverage and len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"blocks must be length {EXPECTED_NUM_BLOCKS}, got {len(blocks)}"
        )
    if not require_full_coverage and len(blocks) < 1:
        raise ValueError("blocks must be non-empty when require_full_coverage=False")

    ids = [int(b["id"]) for b in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"block ids must be unique, got {ids}")
    if require_full_coverage:
        ids_sorted = sorted(ids)
        if ids_sorted != list(range(EXPECTED_NUM_BLOCKS)):
            raise ValueError(
                f"block ids must be 0..{EXPECTED_NUM_BLOCKS - 1} exactly once, got {ids_sorted}"
            )

    shared = cfg.get("shared_zones")
    if not isinstance(shared, list) or len(shared) < 1:
        raise ValueError("shared_zones must be a non-empty list")
    validate_shared_zones_ddim(shared, T)
    nz = len(shared)

    mapped_runtime_names: List[str] = []
    for b in blocks:
        bid = int(b.get("id"))

        mask = b.get("expanded_mask")
        if not isinstance(mask, list) or len(mask) != T:
            raise ValueError(
                f"block id={bid}: expanded_mask length must be T={T}, "
                f"got {len(mask) if isinstance(mask, list) else type(mask)}"
            )
        row = np.asarray(mask, dtype=bool)
        if not bool(row[0]):
            raise ValueError(
                f"block id={bid}: expanded_mask[0] must be True "
                "(step_idx=0 <-> DDPM t=T-1)"
            )

        name = str(b.get("name", ""))
        rt = stage1_block_to_runtime_block(name)
        rid = runtime_name_to_block_id(rt)
        mapped_runtime_names.append(rt)

        rt_declared = b.get("runtime_name", None)
        if rt_declared is not None and str(rt_declared) != rt:
            raise ValueError(
                f"block id={bid}: runtime_name {rt_declared!r} contradicts name {name!r} -> {rt!r}"
            )
        rid_declared = b.get("canonical_runtime_block_id", None)
        if rid_declared is not None and int(rid_declared) != rid:
            raise ValueError(
                f"block id={bid}: canonical_runtime_block_id {rid_declared!r} contradicts name {name!r} -> {rid}"
            )
        local_bid_declared = b.get("scheduler_local_block_id", None)
        if local_bid_declared is not None and int(local_bid_declared) != bid:
            raise ValueError(
                f"block id={bid}: scheduler_local_block_id {local_bid_declared!r} must equal id"
            )

        if require_k_per_zone:
            kz = b.get("k_per_zone")
            if not isinstance(kz, list) or len(kz) != nz:
                raise ValueError(
                    f"block id={bid}: len(k_per_zone) must be {nz}, "
                    f"got {len(kz) if isinstance(kz, list) else type(kz)}"
                )
            kz_int = [int(x) for x in kz]
            for zi, kv in enumerate(kz_int):
                if kv < 1:
                    raise ValueError(f"block id={bid}: k_per_zone[{zi}] must be >= 1, got {kv}")
            rebuilt = np.asarray(
                rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
                    shared,
                    kz_int,
                    T,
                    block_id=bid,
                ),
                dtype=bool,
            )
            if not np.all(row >= rebuilt):
                bad = np.where(~row & rebuilt)[0].tolist()
                raise ValueError(
                    f"block id={bid}: expanded_mask must be >= rebuild(shared_zones,k_per_zone), "
                    f"missing steps {bad[:24]}" + (" ..." if len(bad) > 24 else "")
                )

    if len(set(mapped_runtime_names)) != len(mapped_runtime_names):
        raise ValueError("mapped runtime block names from blocks[].name must be unique")
    if require_full_coverage and set(mapped_runtime_names) != set(RUNTIME_LAYER_NAMES):
        missing = sorted(set(RUNTIME_LAYER_NAMES) - set(mapped_runtime_names))
        extra = sorted(set(mapped_runtime_names) - set(RUNTIME_LAYER_NAMES))
        raise ValueError(
            "mapped runtime block set mismatch: "
            f"missing={missing}, extra={extra}"
        )


def expanded_mask_row_to_recompute_step_indices(
    expanded_mask_row: List[bool] | np.ndarray,
    T: int,
) -> Set[int]:
    """
    單一 block expanded_mask（步序 i）→ recompute 的 step index 集合。
    expanded_mask[i]=True 表示該步需要 full compute。
    """
    row = np.asarray(expanded_mask_row, dtype=bool)
    if row.shape != (T,):
        raise ValueError(f"expanded_mask row must have shape ({T},), got {row.shape}")
    return set(int(i) for i in np.where(row)[0])


def stage1_mask_to_runtime_cache_scheduler(
    cfg: Dict[str, Any],
    *,
    require_k_per_zone: bool = True,
) -> Dict[str, Set[int]]:
    """
    由 Stage1 config 產生完整 runtime cache scheduler。
    key: runtime block 名稱（block_0~block_27）；value: 需 full compute 的 step index 集合。
    """
    validate_stage1_scheduler_config(cfg, require_k_per_zone=require_k_per_zone)
    T = int(cfg["T"])
    blocks = sorted(cfg["blocks"], key=lambda b: int(b["id"]))
    sched: Dict[str, Set[int]] = {}
    for b in blocks:
        rt = stage1_block_to_runtime_block(str(b["name"]))
        if rt in sched:
            raise ValueError(f"duplicate runtime block {rt}")
        sched[rt] = expanded_mask_row_to_recompute_step_indices(b["expanded_mask"], T)
    if len(sched) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"internal error: expected {EXPECTED_NUM_BLOCKS} runtime keys, got {len(sched)}"
        )
    return sched


def cache_scheduler_to_jsonable(sched: Dict[str, Set[int]]) -> Dict[str, List[int]]:
    """set 轉 JSON-friendly sorted list。"""
    return {k: sorted(v) for k, v in sorted(sched.items())}


def prefix_step_indices_first_n(T: int, n: int) -> Set[int]:
    """
    取前 N 個採樣步的 step index 集合（step_idx=0 是第一步）。
    對應 LDM 版的 prefix_ddim_timesteps_first_n，但 DiT 使用 step index 空間。
    """
    if int(n) < 0:
        raise ValueError(f"prefix steps N must be >= 0, got {n!r}")
    if n <= 0:
        return set()
    T = int(T)
    n_eff = min(int(n), T)
    return set(range(0, n_eff))


def apply_cache_scheduler_runtime_overrides(
    sched: Dict[str, Set[int]],
    T: int,
    *,
    force_full_prefix_steps: int = 0,
    force_full_runtime_blocks: Optional[List[str]] = None,
) -> Tuple[Dict[str, Set[int]], Dict[str, Any]]:
    """
    對既有 runtime scheduler 做聯集式保守覆寫（只會增加 recompute，不會減少）。

    - force_full_prefix_steps: 所有 block 前 N 步強制 full（step index 0..N-1）。
    - force_full_runtime_blocks: 指定 block 全 timestep 強制 full。
    """
    T = int(T)
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if int(force_full_prefix_steps) < 0:
        raise ValueError(
            f"force_full_prefix_steps must be >= 0, got {force_full_prefix_steps!r}"
        )
    if set(sched.keys()) != set(RUNTIME_LAYER_NAMES):
        raise ValueError(
            "cache_scheduler keys must match full runtime inventory; "
            f"got {sorted(sched.keys())[:8]}... (len={len(sched)})"
        )

    force_full_runtime_blocks = list(force_full_runtime_blocks or [])
    for name in force_full_runtime_blocks:
        if name not in RUNTIME_LAYER_NAMES:
            raise ValueError(
                f"unknown runtime block in force_full_runtime_blocks: {name!r}"
            )

    out: Dict[str, Set[int]] = {k: set(v) for k, v in sched.items()}
    prefix_idxs = prefix_step_indices_first_n(T, force_full_prefix_steps)
    all_idxs = set(range(T))

    for k in out:
        out[k] |= prefix_idxs
    for name in force_full_runtime_blocks:
        out[name] |= all_idxs

    meta: Dict[str, Any] = {
        "force_full_prefix_steps": int(force_full_prefix_steps),
        "prefix_step_indices": sorted(prefix_idxs),
        "force_full_runtime_blocks": list(force_full_runtime_blocks),
        "note": (
            "Overrides are unions on top of Stage1-expanded recompute sets; "
            "they do not replace Stage1 scheduler JSON."
        ),
    }
    return out, meta
