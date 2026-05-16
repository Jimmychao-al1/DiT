"""
Stage2 (DiT): baseline/cache 兩趟 block residual 收集與誤差統計。

收集每個 runtime block、每個 step index 的 residual tensor（r_msa + r_mlp），
再計算 baseline vs cache 的 L1 / L2 / cosine，並做 block-step / block-zone 聚合。

注意事項：
- 收集的是 residual（r_msa + r_mlp），shape (2B, N_tokens, hidden_dim) = (2B, 256, 1152)
- L1 誤差計算時取 cond half（[:B]），與 Stage 0 的語義一致（cond 分支決定最終生成品質）
- Step index key 對齊 Stage 1 expanded_mask 約定：step_idx=0 → DDPM t=249（第一步）
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from dit_s3cache.stage2.stage2_scheduler_adapter_dit import RUNTIME_LAYER_NAMES

LOGGER = logging.getLogger("Stage2ErrorCollectorDiT")


def _l1_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference（scalar）。"""
    return (a - b).abs().mean()


def _l2_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Root mean squared difference（scalar）。"""
    return (a - b).pow(2).mean().sqrt()


def _flatten_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity after flattening to (N, D)。"""
    a2 = a.flatten(1)
    b2 = b.flatten(1)
    return F.cosine_similarity(a2, b2, dim=1, eps=1e-8)


def _zone_step_indices(shared_zones: List[Dict[str, Any]], T: int) -> Dict[int, List[int]]:
    """
    將 shared_zones 的每個 zone 轉換成 step index 列表。
    zone 的 t_start/t_end 是 DDPM timestep，step_idx = (T-1) - t。
    """
    out: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        ts = int(z["t_start"])
        te = int(z["t_end"])
        # DDPM t 從 ts 降到 te；對應 step_idx 從 (T-1)-ts 升到 (T-1)-te
        step_indices: List[int] = []
        for t in range(te, ts + 1):
            step_idx = (T - 1) - t
            step_indices.append(step_idx)
        out[zid] = step_indices
    return out


class Stage2ErrorCollectorDiT:
    """
    收集 baseline / cache 兩趟每個 block 每個 step 的 residual（r_msa + r_mlp）。

    features 結構：
      feats[run_name][runtime_name][step_idx] = tensor (cpu float, cond half)
    其中 step_idx=0 對應第一步（DDPM t=T-1）。

    L1 誤差計算使用 cond half（[:B]），B = batch_size_per_side。
    """

    def __init__(self, T: int, batch_size_per_side: int, device: Optional[torch.device] = None):
        self.T = int(T)
        self.B = int(batch_size_per_side)  # cond half size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._run_name = "baseline"
        self._debug_write_count = 0
        self._feats: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {
            "baseline": {},
            "cache": {},
        }

    def set_run(self, name: str) -> None:
        """切換收集桶：baseline 或 cache。"""
        if name not in ("baseline", "cache"):
            raise ValueError("run name must be 'baseline' or 'cache'")
        self._run_name = name

    def clear_storage(self) -> None:
        """釋放收集到的 tensor。"""
        self._feats = {"baseline": {}, "cache": {}}
        self._debug_write_count = 0

    def make_residual_callback(self) -> Callable[..., None]:
        """
        產生 callback，供 DiT Stage2 cache wrapper 在每個 block 每個 step 後呼叫。

        callback 簽名：
          cb(runtime_name: str, residual: Tensor, step_idx: int)

        residual shape: (2B, N_tokens, hidden_dim)，儲存前取 cond half（[:B]）。
        """
        return self.on_block_residual

    def on_block_residual(
        self,
        runtime_name: str,
        residual: torch.Tensor,
        step_idx: int,
    ) -> None:
        """
        寫入單一 block-step 的 residual（cond half）。

        runtime_name: 如 "block_0"
        residual: shape (2B, 256, 1152)，自動取 cond half [:B]
        step_idx: 0~T-1，0 = 第一步（DDPM t=T-1）
        """
        if runtime_name not in RUNTIME_LAYER_NAMES:
            LOGGER.warning("[Stage2-DiT] unknown runtime_name=%s; skip", runtime_name)
            return
        if not (0 <= step_idx < self.T):
            LOGGER.warning(
                "[Stage2-DiT] on_block_residual: step_idx=%d out of [0,%d) (block=%s)",
                step_idx,
                self.T,
                runtime_name,
            )
            return

        # 取 cond half
        cond = residual[: self.B].detach().float().cpu()
        bucket = self._feats[self._run_name].setdefault(runtime_name, {})
        bucket[step_idx] = cond
        self._debug_write_count += 1

    def debug_snapshot_line(self, phase: str) -> str:
        """單行摘要，方便 log 追蹤收集進度。"""
        b = self._feats["baseline"]
        c = self._feats["cache"]
        n0b = len(b.get("block_0", {}))
        n0c = len(c.get("block_0", {}))
        return (
            f"[collector debug] phase={phase} id={id(self)} writes={self._debug_write_count} "
            f"baseline_blocks={len(b)} block0_steps={n0b} cache_blocks={len(c)} block0_steps={n0c}"
        )

    def compute_diagnostics(
        self,
        shared_zones: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        將 baseline/cache 收集結果轉為 Stage2 diagnostics dict。

        per_block_step_error key 為字串化 step_idx（對齊 Stage1 expanded_mask 約定）。
        """
        per_block_step_error: Dict[str, Dict[str, Dict[str, float]]] = {}
        per_block_zone_error: Dict[str, Dict[str, Dict[str, Any]]] = {}
        zone_step_idxs = _zone_step_indices(shared_zones, self.T)

        all_l1: List[float] = []
        all_l2: List[float] = []
        all_cos: List[float] = []

        for rt in RUNTIME_LAYER_NAMES:
            per_block_step_error[rt] = {}
            per_block_zone_error[rt] = {}
            bmap = self._feats["baseline"].get(rt, {})
            cmap = self._feats["cache"].get(rt, {})

            for step_idx in range(self.T):
                key = str(step_idx)
                if step_idx not in bmap or step_idx not in cmap:
                    raise RuntimeError(
                        f"[Stage2-DiT] missing residual for {rt} step_idx={step_idx}; "
                        f"baseline keys={sorted(bmap.keys())[:8]}..., "
                        f"cache keys={sorted(cmap.keys())[:8]}..."
                    )
                a = bmap[step_idx]
                b = cmap[step_idx]
                if a.shape != b.shape:
                    raise RuntimeError(
                        f"{rt} step_idx={step_idx}: shape mismatch {a.shape} vs {b.shape}"
                    )

                l1 = float(_l1_scalar(a, b))
                l2 = float(_l2_rms(a, b))
                cos = float(_flatten_cosine(a, b).mean())
                per_block_step_error[rt][key] = {"l1": l1, "l2": l2, "cosine": cos}
                all_l1.append(l1)
                all_l2.append(l2)
                all_cos.append(cos)

            for zid, step_indices in zone_step_idxs.items():
                zs_l1: List[float] = []
                zs_l2: List[float] = []
                zs_cos: List[float] = []
                for si in step_indices:
                    st = per_block_step_error[rt].get(str(si))
                    if st is None:
                        continue
                    zs_l1.append(float(st["l1"]))
                    zs_l2.append(float(st["l2"]))
                    zs_cos.append(float(st["cosine"]))
                if not zs_l1:
                    per_block_zone_error[rt][str(zid)] = {
                        "mean_l1": float("nan"),
                        "mean_l2": float("nan"),
                        "mean_cosine": float("nan"),
                        "num_steps": len(step_indices),
                        "num_compared_in_zone": 0,
                    }
                else:
                    per_block_zone_error[rt][str(zid)] = {
                        "mean_l1": float(np.mean(zs_l1)),
                        "mean_l2": float(np.mean(zs_l2)),
                        "mean_cosine": float(np.mean(zs_cos)),
                        "num_steps": len(step_indices),
                        "num_compared_in_zone": len(zs_l1),
                    }

        global_summary = {
            "mean_l1": float(np.mean(all_l1)) if all_l1 else None,
            "mean_l2": float(np.mean(all_l2)) if all_l2 else None,
            "mean_cosine": float(np.mean(all_cos)) if all_cos else None,
            "num_entries": len(all_l1),
            "cfg_half_note": "L1 computed on cond half ([:B]) only; uncond half excluded.",
        }

        return {
            "per_block_step_error": per_block_step_error,
            "per_block_zone_error": per_block_zone_error,
            "global_summary": global_summary,
            "T": self.T,
            "batch_size_per_side": self.B,
            "time_axis_note": (
                "per_block_step_error key 為字串化 step_idx（0=第一步 DDPM t=249，249=最後步 t=0）。"
                "對齊 Stage1 expanded_mask 約定。"
            ),
        }


def aggregate_per_step_index(
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """對所有 block 聚合每個 step index 的 mean/max L1。"""
    by_si: Dict[str, List[float]] = {}
    for _rt, steps in per_block_step_error.items():
        for si_str, m in steps.items():
            by_si.setdefault(si_str, []).append(float(m["l1"]))

    out: Dict[str, Dict[str, float]] = {}
    for si_str, vals in by_si.items():
        out[si_str] = {
            "mean_l1": float(np.mean(vals)),
            "max_l1": float(np.max(vals)),
        }
    return out
