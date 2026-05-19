"""
Stage2 scheduler JSON → DiT ``cache_runner`` 用的 recompute 排程（model raw timestep）。

``CachedDiTBlock`` 以 ``t[0].item()`` 的 **SpacedDiffusion model timestep**（raw_t）判斷是否
recompute；Stage1/2 的 ``expanded_mask`` 則以 **步序 step_idx**（0=第一步）索引。

轉換規則（與 ``fid_cache_sensitivity`` / ``create_dit_cache_config`` 一致）::

    sampling_timesteps = list(reversed(diffusion.timestep_map))[:T]

對每個 block、每個 step_idx，若 ``expanded_mask[step_idx]`` 為 True，則將
``sampling_timesteps[step_idx]`` 加入該 block 的 recompute set。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

from dit_s3cache.stage2.stage2_scheduler_adapter_dit import (
    DIT_NUM_BLOCKS,
    EXPECTED_NUM_BLOCKS,
    load_stage1_scheduler_config,
    stage1_block_to_runtime_block,
    validate_stage1_scheduler_config,
    validate_time_order,
)


def assert_injective_sampling_timesteps(sampling_timesteps: List[int], *, label: str = "sampling_timesteps") -> None:
    """確認步序→raw_t 一對一，否則 True 格數可能大於 len(recompute_set)。"""
    if len(sampling_timesteps) != len(set(sampling_timesteps)):
        raise ValueError(
            f"{label}: duplicate model timesteps collapse recompute indices; "
            f"unique count={len(set(sampling_timesteps))} vs len={len(sampling_timesteps)}"
        )


def stage2_json_to_dit_cache_scheduler(
    cfg: Dict[str, Any],
    sampling_timesteps: List[int],
    *,
    validate_cfg: bool = True,
    require_injective_timesteps: bool = True,
) -> Dict[int, Set[int]]:
    """
    將 Stage1/Stage2 refined scheduler JSON 轉成 ``cache_runner.CacheScheduler``.

    Args:
        cfg: 已 parse 的 scheduler dict（須含 T, time_order, blocks[].expanded_mask）。
        sampling_timesteps: 長度 T，denoising 順序的 model timestep 列表，
            應為 ``list(reversed(diffusion.timestep_map))[:T]``。
        validate_cfg: 是否呼叫 ``validate_stage1_scheduler_config``。
        require_injective_timesteps: 是否要求 timestep 不重複（建議開）。

    Returns:
        ``{block_idx: set(raw_t,...)}``，共 28 個 key（0..27）。
        集合內為需 **full compute / recompute** 的 raw timestep。
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    T = int(cfg["T"])
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    validate_time_order(str(cfg.get("time_order")), T)

    if len(sampling_timesteps) != T:
        raise ValueError(
            f"sampling_timesteps length {len(sampling_timesteps)} != cfg T={T}"
        )
    if require_injective_timesteps:
        assert_injective_sampling_timesteps(list(sampling_timesteps))

    if validate_cfg:
        validate_stage1_scheduler_config(cfg)

    blocks = sorted(cfg.get("blocks", []), key=lambda b: int(b["id"]))
    if len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"expected {EXPECTED_NUM_BLOCKS} blocks, got {len(blocks)}"
        )

    out: Dict[int, Set[int]] = {}
    st_list = [int(x) for x in sampling_timesteps]

    for b in blocks:
        bid = int(b["id"])
        if bid < 0 or bid >= DIT_NUM_BLOCKS:
            raise ValueError(f"invalid block id {bid}")
        name = str(b.get("name", ""))
        rt = stage1_block_to_runtime_block(name)
        expected_rt = f"block_{bid}"
        if rt != expected_rt:
            raise ValueError(
                f"block id={bid}: name {name!r} maps to {rt!r}, expected {expected_rt!r}"
            )

        mask = b.get("expanded_mask")
        if not isinstance(mask, list) or len(mask) != T:
            raise ValueError(
                f"block {bid}: expanded_mask must be list of length T={T}"
            )
        row = np.asarray(mask, dtype=bool)
        if row.shape != (T,):
            raise ValueError(f"block {bid}: bad expanded_mask shape")

        recompute: Set[int] = set()
        for step_idx, full in enumerate(mask):
            if bool(full):
                recompute.add(st_list[step_idx])

        n_true = int(row.sum())
        if len(recompute) != n_true:
            raise RuntimeError(
                f"block {bid}: |recompute set|={len(recompute)} != expanded_mask True count={n_true} "
                "(injectivity should have been checked)"
            )
        out[bid] = recompute

    if set(out.keys()) != set(range(EXPECTED_NUM_BLOCKS)):
        raise ValueError(
            f"block id set must be 0..{EXPECTED_NUM_BLOCKS - 1}, got {sorted(out.keys())}"
        )
    return out


def full_compute_dit_cache_scheduler(
    sampling_timesteps: List[int],
    *,
    n_blocks: int = EXPECTED_NUM_BLOCKS,
    require_injective_timesteps: bool = True,
) -> Dict[int, Set[int]]:
    """Baseline：每個 block 在每一步都 recompute（等同無 cache reuse）。"""
    if n_blocks != EXPECTED_NUM_BLOCKS:
        raise ValueError(f"n_blocks must be {EXPECTED_NUM_BLOCKS}, got {n_blocks}")
    if require_injective_timesteps:
        assert_injective_sampling_timesteps(list(sampling_timesteps))
    full = set(int(x) for x in sampling_timesteps)
    if len(full) != len(sampling_timesteps):
        raise ValueError("injectivity violated")
    return {i: set(full) for i in range(n_blocks)}


def load_scheduler_and_build_cache_scheduler(
    json_path: str | Path,
    sampling_timesteps: List[int],
    **kwargs: Any,
) -> Dict[int, Set[int]]:
    """從磁盘加载 JSON 并转换。"""
    cfg = load_stage1_scheduler_config(json_path)
    return stage2_json_to_dit_cache_scheduler(cfg, sampling_timesteps, **kwargs)


__all__ = [
    "assert_injective_sampling_timesteps",
    "stage2_json_to_dit_cache_scheduler",
    "full_compute_dit_cache_scheduler",
    "load_scheduler_and_build_cache_scheduler",
]
