"""
驗證 Stage2-DiT 產物：
1) refined scheduler config JSON
2) blockwise threshold config JSON
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _bootstrap_local_paths() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_bootstrap_local_paths()

from dit_s3cache.stage1.stage1_scheduler_dit import validate_shared_zones_ddim
from dit_s3cache.stage2.stage2_scheduler_adapter_dit import (
    EXPECTED_NUM_BLOCKS,
    RUNTIME_LAYER_NAMES,
    TIME_ORDER_EXPECTED,
    rebuild_expanded_mask_from_shared_zones_and_k_per_zone,
    runtime_block_to_stage1_name,
    stage1_block_to_runtime_block,
)


def verify_refined_scheduler_config(
    cfg: Dict[str, Any],
    *,
    require_full_coverage: bool = True,
    full_compute_ratio_range: tuple[float, float] = (0.05, 1.0),
) -> None:
    """
    驗證 stage2_refined_scheduler_config.json。

    額外檢查（DiT 版新增）：
    - full_compute_ratio 在合理範圍 full_compute_ratio_range
    """
    if cfg.get("time_order") != TIME_ORDER_EXPECTED:
        raise ValueError(
            f"time_order must be {TIME_ORDER_EXPECTED!r}, got {cfg.get('time_order')!r}"
        )

    T = int(cfg["T"])
    if T < 2:
        raise ValueError(f"T must be >= 2, got {T}")

    shared = cfg.get("shared_zones")
    if not isinstance(shared, list):
        raise TypeError("shared_zones must be a list")
    validate_shared_zones_ddim(shared, T)
    nz = len(shared)

    blocks = cfg.get("blocks")
    if not isinstance(blocks, list):
        raise TypeError("blocks must be a list")
    if require_full_coverage and len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(f"blocks must have length {EXPECTED_NUM_BLOCKS}, got {len(blocks)}")
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

    mapped_runtime_names: List[str] = []
    total_full = 0
    total_cells = 0

    for b in blocks:
        bid = int(b["id"])
        name = str(b.get("name", ""))
        rt = stage1_block_to_runtime_block(name)
        mapped_runtime_names.append(rt)
        rid = RUNTIME_LAYER_NAMES.index(rt)

        local_bid_declared = b.get("scheduler_local_block_id", None)
        if local_bid_declared is not None and int(local_bid_declared) != bid:
            raise ValueError(
                f"block id={bid}: scheduler_local_block_id {local_bid_declared!r} must equal id"
            )
        rid_declared = b.get("canonical_runtime_block_id", None)
        if rid_declared is not None and int(rid_declared) != rid:
            raise ValueError(
                f"block id={bid}: canonical_runtime_block_id {rid_declared!r} contradicts name {name!r} -> {rid}"
            )
        rt_declared = b.get("runtime_name", None)
        if rt_declared is not None and str(rt_declared) != rt:
            raise ValueError(
                f"block id={bid}: runtime_name {rt_declared!r} contradicts name {name!r} -> {rt!r}"
            )

    if len(set(mapped_runtime_names)) != len(mapped_runtime_names):
        raise ValueError("mapped runtime names from blocks[].name must be unique")
    if require_full_coverage:
        if set(mapped_runtime_names) != set(RUNTIME_LAYER_NAMES):
            missing = sorted(set(RUNTIME_LAYER_NAMES) - set(mapped_runtime_names))
            extra = sorted(set(mapped_runtime_names) - set(RUNTIME_LAYER_NAMES))
            raise ValueError(
                "mapped runtime block set mismatch for full config: "
                f"missing={missing}, extra={extra}"
            )

    for b in blocks:
        bid = int(b["id"])
        em = b.get("expanded_mask")
        if not isinstance(em, list) or len(em) != T:
            raise ValueError(
                f"block id={bid}: expanded_mask must be length T={T}, "
                f"got {len(em) if isinstance(em, list) else type(em)}"
            )
        row = np.asarray(em, dtype=bool)
        if row.shape != (T,):
            raise ValueError(f"block id={bid}: bad expanded_mask shape")
        if not bool(row[0]):
            raise ValueError(
                f"block id={bid}: expanded_mask[0] must be True (step_idx=0 <-> DDPM t=T-1)"
            )
        # 確認只含 True/False（bool 型別）
        for i, v in enumerate(em):
            if not isinstance(v, bool):
                raise TypeError(
                    f"block id={bid}: expanded_mask[{i}] must be bool, got {type(v).__name__}"
                )

        kz = b.get("k_per_zone")
        if not isinstance(kz, list):
            raise TypeError(f"block id={bid}: k_per_zone must be a list")
        if len(kz) != nz:
            raise ValueError(f"block id={bid}: len(k_per_zone)={len(kz)} != len(shared_zones)={nz}")
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
                f"block id={bid}: expanded_mask must be >= mask from k_per_zone (missing steps {bad[:24]}"
                + (" ..." if len(bad) > 24 else "")
                + ")"
            )

        total_full += int(row.sum())
        total_cells += T

    if total_cells > 0:
        ratio = total_full / total_cells
        lo, hi = full_compute_ratio_range
        if not (lo <= ratio <= hi):
            raise ValueError(
                f"full_compute_ratio={ratio:.4f} is outside expected range [{lo}, {hi}]. "
                "This may indicate an overly aggressive or degenerate scheduler."
            )


def verify_blockwise_threshold_config_dict(
    data: Dict[str, Any],
    *,
    eps: float = 1e-9,
) -> None:
    """
    驗證 stage2_thresholds_blockwise.json。

    檢查：
    - per_block 長度 = 28（DiT 版）
    - block_id 完整覆蓋 0..27
    - zone/peak threshold 均為正且有限
    - peak >= ratio_min * zone
    - runtime_name 對應 block_X
    """
    if not isinstance(data, dict):
        raise TypeError("threshold config root must be a dict")
    method = data.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("threshold config must have non-empty string 'method'")

    ratio_min = float(data.get("peak_over_zone_ratio_min", 0.0))
    if not (ratio_min > 0.0) or math.isnan(ratio_min) or math.isinf(ratio_min):
        raise ValueError(
            f"peak_over_zone_ratio_min must be finite and > 0, got {ratio_min!r}"
        )

    blocks = data.get("per_block")
    if not isinstance(blocks, list) or len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"per_block must be a list of length {EXPECTED_NUM_BLOCKS}, "
            f"got {len(blocks) if isinstance(blocks, list) else type(blocks)}"
        )

    seen_ids: set[int] = set()
    for entry in blocks:
        if not isinstance(entry, dict):
            raise TypeError("each per_block entry must be a dict")
        bid = int(entry["block_id"])
        if bid < 0 or bid >= EXPECTED_NUM_BLOCKS:
            raise ValueError(f"invalid block_id {bid}")
        if bid in seen_ids:
            raise ValueError(f"duplicate block_id {bid}")
        seen_ids.add(bid)

        # DiT block: runtime_name == canonical_name == "block_X"
        rt = str(entry.get("runtime_name", ""))
        if rt != RUNTIME_LAYER_NAMES[bid]:
            raise ValueError(
                f"block_id {bid}: runtime_name must be {RUNTIME_LAYER_NAMES[bid]!r}, got {rt!r}"
            )
        rid = entry.get("canonical_runtime_block_id", None)
        if rid is not None and int(rid) != bid:
            raise ValueError(
                f"block_id {bid}: canonical_runtime_block_id must equal block_id, got {rid!r}"
            )
        canonical_name = str(entry.get("canonical_name", ""))
        if canonical_name != f"block_{bid}":
            raise ValueError(
                f"block_id {bid}: canonical_name must be 'block_{bid}', got {canonical_name!r}"
            )

        zt = float(entry["zone_l1_threshold"])
        pt = float(entry["peak_l1_threshold"])
        if math.isnan(zt) or math.isinf(zt) or zt <= 0.0:
            raise ValueError(
                f"block_id {bid}: zone_l1_threshold must be finite and > 0, got {zt!r}"
            )
        if math.isnan(pt) or math.isinf(pt) or pt <= 0.0:
            raise ValueError(
                f"block_id {bid}: peak_l1_threshold must be finite and > 0, got {pt!r}"
            )
        if pt + eps < ratio_min * zt:
            raise ValueError(
                f"block_id {bid}: need peak_l1_threshold >= peak_over_zone_ratio_min * zone_l1_threshold "
                f"({pt} < {ratio_min} * {zt})"
            )

    if seen_ids != set(range(EXPECTED_NUM_BLOCKS)):
        raise ValueError(
            f"block_id set must be 0..{EXPECTED_NUM_BLOCKS - 1} exactly once, got {sorted(seen_ids)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify Stage2-DiT refined scheduler config or blockwise threshold config."
    )
    ap.add_argument(
        "config_path",
        type=str,
        nargs="?",
        default=None,
        help="stage2_refined_scheduler_config.json (or Stage1-compatible config)",
    )
    ap.add_argument(
        "--threshold-config",
        type=str,
        default=None,
        help="If provided, validate stage2_thresholds_blockwise.json instead",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial scheduler (not enforcing full 28-block coverage)",
    )
    ap.add_argument(
        "--num-blocks",
        type=int,
        default=EXPECTED_NUM_BLOCKS,
        help=f"Expected number of blocks (default {EXPECTED_NUM_BLOCKS})",
    )
    ap.add_argument(
        "--num-steps",
        type=int,
        default=250,
        help="Expected T (default 250)",
    )
    args = ap.parse_args()

    if args.threshold_config:
        p = Path(args.threshold_config)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        verify_blockwise_threshold_config_dict(data)
        print(f"OK (threshold config): {p.resolve()}")
        return

    if not args.config_path:
        ap.error("Need config_path, or use --threshold-config")

    p = Path(args.config_path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    verify_refined_scheduler_config(cfg, require_full_coverage=not bool(args.allow_partial))
    print(f"OK: {p.resolve()}")


if __name__ == "__main__":
    main()
