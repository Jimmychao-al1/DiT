"""
Stage2 (DiT) 主流程：
- 讀入 Stage1 scheduler_config.json
- baseline / cache 兩趟採樣並收集每個 DiTBlock 的 residual 誤差
- 單輪 refinement（zone 降 k + peak 補 mask）
- 輸出 refined scheduler 與診斷 JSON

用法：
  Pass 1（global refine）:
    python stage2_runtime_refine_dit.py \\
        --scheduler-config ... --threshold-mode global --output-dir .../00_global_refine

  Pass 2（blockwise refine）:
    python stage2_runtime_refine_dit.py \\
        --scheduler-config ... --threshold-mode blockwise \\
        --threshold-config .../01_blockwise_threshold/stage2_thresholds_blockwise.json \\
        --output-dir .../02_refined_blockwise
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from download import find_model
from models import DiT_models, modulate

from diffusion import create_diffusion

from dit_s3cache.stage2.stage2_error_collector_dit import (
    Stage2ErrorCollectorDiT,
    aggregate_per_step_index,
)
from dit_s3cache.stage2.stage2_scheduler_adapter_dit import (
    EXPECTED_NUM_BLOCKS,
    RUNTIME_LAYER_NAMES,
    apply_cache_scheduler_runtime_overrides,
    cache_scheduler_to_jsonable,
    ddpm_t_to_step_index,
    load_stage1_scheduler_config,
    parse_time_order,
    rebuild_expanded_mask_from_shared_zones_and_k_per_zone,
    runtime_name_to_block_id,
    stage1_block_to_runtime_block,
    stage1_mask_to_runtime_cache_scheduler,
    step_index_to_ddpm_t,
    validate_stage1_scheduler_config,
)
from dit_s3cache.stage2.verify_stage2_dit import (
    verify_blockwise_threshold_config_dict,
    verify_refined_scheduler_config,
)

LOGGER = logging.getLogger("Stage2RuntimeRefineDiT")
_STAGE2_DIR = Path(__file__).resolve().parent
_STAGE2_LOG_FILE = _STAGE2_DIR / "stage2_runtime_refine_dit.log"
_LOG_FMT = logging.Formatter("%(asctime)s [%(levelname)s] [Stage2-DiT] %(message)s")

# 預設 evaluation 參數
DEFAULT_EVAL_NUM_IMAGES = 8
DEFAULT_EVAL_CHUNK_SIZE = 1

# 預設 global threshold（參考 LDM baseline）
DEFAULT_ZONE_L1_THRESHOLD = 0.02
DEFAULT_PEAK_L1_THRESHOLD = 0.08

# DiT 模型預設值
DEFAULT_DIT_MODEL = "DiT-XL/2"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_NUM_CLASSES = 1000
DEFAULT_CFG_SCALE = 1.5
DEFAULT_NUM_SAMPLING_STEPS = 250

# 固定 class label（error collection reproducibility）
FIXED_EVAL_CLASSES = list(range(1000))  # 使用 class 0~999 循環取前 N 個


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_stage2_logging() -> None:
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)

    h_err = logging.StreamHandler(sys.stderr)
    h_err.setFormatter(_LOG_FMT)
    LOGGER.addHandler(h_err)

    h_file = logging.FileHandler(_STAGE2_LOG_FILE, mode="a", encoding="utf-8")
    h_file.setFormatter(_LOG_FMT)
    LOGGER.addHandler(h_file)

    LOGGER.propagate = False


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(x) for x in obj)
    if isinstance(obj, np.floating):
        return _json_safe(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# DiT Stage2 Cache Wrapper
# ---------------------------------------------------------------------------


class DiTStage2Block:
    """
    Monkey-patch 單一 DiTBlock，同時支援：
    1. Baseline 模式（cache_enabled=False）：全 full compute，收集 residual
    2. Cache 模式（cache_enabled=True）：按 recompute_step_indices 決定 recompute 或 reuse

    callback 簽名：
      callback(runtime_name: str, residual: torch.Tensor, step_idx: int)
      residual shape: (2B, N_tokens, hidden_dim)
    """

    def __init__(
        self,
        block: torch.nn.Module,
        block_idx: int,
        runtime_name: str,
        recompute_step_indices: Set[int],
        raw_t_to_step_idx: Dict[int, int],
        callback: Optional[Callable],
        cache_enabled: bool,
    ) -> None:
        self.block = block
        self.block_idx = block_idx
        self.runtime_name = runtime_name
        self.recompute_step_indices = set(recompute_step_indices)
        self.raw_t_to_step_idx = dict(raw_t_to_step_idx)
        self.callback = callback
        self.cache_enabled = bool(cache_enabled)

        self.cached_residual: Optional[torch.Tensor] = None
        self.current_raw_t: Optional[int] = None  # 由外部在每步 forward_with_cfg 前設定
        self.cache_hits = 0
        self.recompute_hits = 0

        self._original_forward: Callable = block.forward
        self._is_installed = False
        self.install()

    def install(self) -> None:
        if self._is_installed:
            return
        block = self.block

        def hooked_forward(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            raw_t = self.current_raw_t
            if raw_t is None:
                raise RuntimeError(
                    f"DiTStage2Block [{self.runtime_name}]: current_raw_t not set. "
                    "Ensure forward_with_cfg wrapper broadcasts timestep before each pass."
                )
            step_idx = self.raw_t_to_step_idx.get(int(raw_t))
            if step_idx is None:
                raise RuntimeError(
                    f"DiTStage2Block [{self.runtime_name}]: raw_t={raw_t} not in raw_t_to_step_idx"
                )

            should_recompute = (
                not self.cache_enabled
                or (step_idx in self.recompute_step_indices)
                or (self.cached_residual is None)
            )

            if should_recompute:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    block.adaLN_modulation(c).chunk(6, dim=1)
                )
                r_msa = gate_msa.unsqueeze(1) * block.attn(
                    modulate(block.norm1(x), shift_msa, scale_msa)
                )
                x_after_msa = x + r_msa
                r_mlp = gate_mlp.unsqueeze(1) * block.mlp(
                    modulate(block.norm2(x_after_msa), shift_mlp, scale_mlp)
                )
                residual = r_msa + r_mlp
                if self.cache_enabled:
                    self.cached_residual = residual.detach()
                self.recompute_hits += 1
                output = x_after_msa + r_mlp
            else:
                residual = self.cached_residual
                output = x + residual
                self.cache_hits += 1

            if self.callback is not None:
                self.callback(self.runtime_name, residual.detach(), step_idx)

            return output

        self.block.forward = hooked_forward
        self._is_installed = True

    def reset_cache(self) -> None:
        self.cached_residual = None
        self.current_raw_t = None

    def restore(self) -> None:
        if not self._is_installed:
            return
        self.block.forward = self._original_forward
        self._is_installed = False
        self.reset_cache()

    def stats(self) -> Dict[str, int]:
        return {
            "cache_hits": int(self.cache_hits),
            "recompute_hits": int(self.recompute_hits),
            "total_hook_calls": int(self.cache_hits + self.recompute_hits),
        }


class DiTStage2Context:
    """
    Context manager：安裝全部 28 個 DiTBlock 的 Stage2 patch，
    並包裝 model.forward_with_cfg 以廣播當前 timestep。
    退出時自動還原所有 patch。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        recompute_step_indices_by_block: Dict[str, Set[int]],
        raw_t_to_step_idx: Dict[int, int],
        callback: Optional[Callable],
        cache_enabled: bool,
        T: Optional[int] = None,
    ) -> None:
        self.model = model
        self.recompute_by_block = {k: set(v) for k, v in recompute_step_indices_by_block.items()}
        self.raw_t_to_step_idx = dict(raw_t_to_step_idx)
        self.callback = callback
        self.cache_enabled = bool(cache_enabled)
        self.T = int(T if T is not None else len(raw_t_to_step_idx))

        self._stage2_blocks: List[DiTStage2Block] = []
        self._original_forward_with_cfg: Optional[Callable] = None

    def __enter__(self) -> "DiTStage2Context":
        # 安裝每個 DiTBlock 的 Stage2 patch
        for block_idx, block in enumerate(self.model.blocks):
            rt = f"block_{block_idx}"
            if rt not in RUNTIME_LAYER_NAMES:
                raise ValueError(f"unexpected block index {block_idx} (max={EXPECTED_NUM_BLOCKS - 1})")
            recompute_steps = self.recompute_by_block.get(rt, set(range(self.T)))
            stage2_block = DiTStage2Block(
                block=block,
                block_idx=block_idx,
                runtime_name=rt,
                recompute_step_indices=recompute_steps,
                raw_t_to_step_idx=self.raw_t_to_step_idx,
                callback=self.callback,
                cache_enabled=self.cache_enabled,
            )
            self._stage2_blocks.append(stage2_block)

        # 包裝 forward_with_cfg 以廣播 raw_t 給每個 block
        original_fwcfg = self.model.forward_with_cfg
        stage2_blocks = self._stage2_blocks

        def _wrapped_forward_with_cfg(
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            cfg_scale: float,
        ) -> torch.Tensor:
            raw_t = int(t[0].item())
            for blk in stage2_blocks:
                blk.current_raw_t = raw_t
            return original_fwcfg(x, t, y, cfg_scale)

        self._original_forward_with_cfg = original_fwcfg
        self.model.forward_with_cfg = _wrapped_forward_with_cfg

        LOGGER.info(
            "DiTStage2Context enabled | cache_enabled=%s | blocks=%d",
            self.cache_enabled,
            len(self._stage2_blocks),
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        for blk in self._stage2_blocks:
            blk.restore()
        self._stage2_blocks.clear()
        if self._original_forward_with_cfg is not None:
            self.model.forward_with_cfg = self._original_forward_with_cfg
            self._original_forward_with_cfg = None

    def reset_all_cache(self) -> None:
        for blk in self._stage2_blocks:
            blk.reset_cache()

    def aggregate_stats(self) -> Dict[str, Any]:
        total_recompute = sum(b.recompute_hits for b in self._stage2_blocks)
        total_cache = sum(b.cache_hits for b in self._stage2_blocks)
        return {
            "recompute_hits": total_recompute,
            "cache_hits": total_cache,
            "total_hook_calls": total_recompute + total_cache,
            "per_block": {b.runtime_name: b.stats() for b in self._stage2_blocks},
        }


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def _run_single_sampling_pass(
    *,
    model: torch.nn.Module,
    diffusion: Any,
    z_T: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float,
    device: torch.device,
    sampler: str = "ddpm",
    eta: float = 0.0,
) -> None:
    """執行一次完整 sampling loop（progressive，用於 hook side-effects）。"""
    model_kwargs = {"y": y, "cfg_scale": cfg_scale}
    sample_fn = (
        diffusion.ddim_sample_loop_progressive
        if sampler == "ddim"
        else diffusion.p_sample_loop_progressive
    )
    sample_kwargs = dict(
        model=model.forward_with_cfg,
        shape=z_T.shape,
        noise=z_T,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
    )
    if sampler == "ddim":
        sample_kwargs["eta"] = float(eta)
    sample_iter = sample_fn(**sample_kwargs)
    for _out in sample_iter:
        pass  # 只需要 hook side-effects（殘差收集），不需要輸出


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_step_metrics_inplace(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
    *,
    batch_size: int,
) -> None:
    w = float(batch_size)
    for rt, steps in per_block_step_error.items():
        rt_acc = agg.setdefault(rt, {})
        for si_str, m in steps.items():
            slot = rt_acc.setdefault(
                si_str,
                {"sum_l1": 0.0, "sum_l2_sq": 0.0, "sum_cos": 0.0, "weight": 0.0},
            )
            l1 = float(m["l1"])
            l2 = float(m["l2"])
            cos = float(m["cosine"])
            slot["sum_l1"] += l1 * w
            slot["sum_l2_sq"] += (l2 * l2) * w
            slot["sum_cos"] += cos * w
            slot["weight"] += w


def _finalize_per_block_step_error(
    agg: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rt, steps in agg.items():
        row: Dict[str, Dict[str, float]] = {}
        for si_str, slot in steps.items():
            w = float(slot["weight"])
            if w <= 0.0:
                continue
            row[si_str] = {
                "l1": float(slot["sum_l1"] / w),
                "l2": float(math.sqrt(max(slot["sum_l2_sq"] / w, 0.0))),
                "cosine": float(slot["sum_cos"] / w),
            }
        out[rt] = row
    return out


def _compute_per_block_zone_error(
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
    shared_zones: List[Dict[str, Any]],
    T: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """重新計算 zone-level error（在 aggregation 之後）。"""
    per_block_zone_error: Dict[str, Dict[str, Dict[str, Any]]] = {}
    zone_step_idxs: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        ts = int(z["t_start"])
        te = int(z["t_end"])
        step_indices: List[int] = []
        for t in range(te, ts + 1):
            step_indices.append((T - 1) - t)
        zone_step_idxs[zid] = step_indices

    for rt, steps in per_block_step_error.items():
        per_block_zone_error[rt] = {}
        for zid, step_indices in zone_step_idxs.items():
            zs_l1: List[float] = []
            zs_l2: List[float] = []
            zs_cos: List[float] = []
            for si in step_indices:
                st = steps.get(str(si))
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
    return per_block_zone_error


def _build_diagnostics(
    *,
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
    shared_zones: List[Dict[str, Any]],
    T: int,
) -> Dict[str, Any]:
    per_block_zone_error = _compute_per_block_zone_error(per_block_step_error, shared_zones, T)
    all_l1: List[float] = []
    all_l2: List[float] = []
    all_cos: List[float] = []
    for steps in per_block_step_error.values():
        for m in steps.values():
            all_l1.append(float(m["l1"]))
            all_l2.append(float(m["l2"]))
            all_cos.append(float(m["cosine"]))

    global_summary = {
        "mean_l1": float(np.mean(all_l1)) if all_l1 else None,
        "mean_l2": float(np.mean(all_l2)) if all_l2 else None,
        "mean_cosine": float(np.mean(all_cos)) if all_cos else None,
        "num_entries": len(all_l1),
        "cfg_half_note": "L1 computed on cond half ([:B]) of residual (r_msa + r_mlp).",
    }

    return {
        "per_block_step_error": per_block_step_error,
        "per_block_zone_error": per_block_zone_error,
        "global_summary": global_summary,
        "T": int(T),
        "time_axis_note": (
            "per_block_step_error key 為字串化 step_idx（0=第一步 DDPM t=249，"
            "249=最後步 t=0）。對齊 Stage1 expanded_mask 約定。"
        ),
    }


# ---------------------------------------------------------------------------
# Blockwise threshold loader
# ---------------------------------------------------------------------------


def _load_blockwise_threshold_config(
    path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """讀取 blockwise threshold JSON；回傳 (runtime_name → entry, block_id → entry, root)。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"threshold config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    verify_blockwise_threshold_config_dict(data)

    by_runtime: Dict[str, Dict[str, Any]] = {}
    by_id: Dict[int, Dict[str, Any]] = {}
    for entry in data["per_block"]:
        bid = int(entry["block_id"])
        rt = str(entry["runtime_name"])
        if rt in by_runtime:
            raise ValueError(f"duplicate runtime_name in threshold config: {rt}")
        by_runtime[rt] = entry
        by_id[bid] = entry

    if len(by_runtime) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"threshold config must contain exactly {EXPECTED_NUM_BLOCKS} runtime entries, got {len(by_runtime)}"
        )
    return by_runtime, by_id, data


# ---------------------------------------------------------------------------
# Main refine function
# ---------------------------------------------------------------------------


def run_stage2_refine_dit(
    *,
    scheduler_config_path: str,
    output_dir: str,
    seed: int = 0,
    zone_l1_threshold: float = DEFAULT_ZONE_L1_THRESHOLD,
    peak_l1_threshold: float = DEFAULT_PEAK_L1_THRESHOLD,
    threshold_config_path: Optional[str] = None,
    model_name: str = DEFAULT_DIT_MODEL,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    num_sampling_steps: int = DEFAULT_NUM_SAMPLING_STEPS,
    sampler: str = "ddpm",
    eta: float = 0.0,
    ckpt: Optional[str] = None,
    device: Optional[torch.device] = None,
    eval_num_images: int = DEFAULT_EVAL_NUM_IMAGES,
    eval_chunk_size: int = DEFAULT_EVAL_CHUNK_SIZE,
    force_full_prefix_steps: int = 0,
    force_full_runtime_blocks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    執行 Stage2 單趟 refinement（Pass 1 or Pass 2 通用）。

    - threshold_config_path=None  → global threshold mode
    - threshold_config_path=<path> → blockwise threshold mode
    """
    _configure_stage2_logging()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage2-DiT runtime refine.")

    device = device or torch.device("cuda")
    _seed_all(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Stage1 config ---
    cfg = load_stage1_scheduler_config(scheduler_config_path)
    validate_stage1_scheduler_config(cfg)
    T = int(cfg["T"])
    cfg_sampler, _ = parse_time_order(str(cfg.get("time_order")))
    sampler = str(sampler).lower()
    if sampler not in {"ddpm", "ddim"}:
        raise ValueError(f"sampler must be 'ddpm' or 'ddim', got {sampler!r}")
    if sampler != cfg_sampler:
        raise ValueError(
            f"--sampler {sampler!r} does not match scheduler time_order sampler {cfg_sampler!r}"
        )
    shared_zones: List[Dict[str, Any]] = cfg["shared_zones"]

    # --- Build runtime cache scheduler (step_index space) ---
    cache_sched_stage1 = stage1_mask_to_runtime_cache_scheduler(cfg)

    blocks_eff = list(force_full_runtime_blocks or [])
    cache_sched_effective, override_meta = apply_cache_scheduler_runtime_overrides(
        cache_sched_stage1,
        T,
        force_full_prefix_steps=int(force_full_prefix_steps),
        force_full_runtime_blocks=blocks_eff,
    )

    # --- Load DiT model ---
    LOGGER.info("Loading DiT model: %s (image_size=%d)", model_name, image_size)
    latent_size = image_size // 8
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
    ).to(device)
    ckpt_path = ckpt or f"DiT-XL-2-{image_size}x{image_size}.pt"
    model.load_state_dict(find_model(ckpt_path))
    model.eval()
    LOGGER.info("Model loaded.")

    # --- Build diffusion + timestep map ---
    diffusion = create_diffusion(str(num_sampling_steps))
    # timestep_map[step_idx] = raw_t（model 實際看到的 noise schedule timestep）
    timestep_map_reversed = list(reversed(diffusion.timestep_map))[:T]
    if len(timestep_map_reversed) != T:
        raise RuntimeError(
            f"Unexpected timestep map length: got {len(timestep_map_reversed)}, expected T={T}"
        )
    raw_t_to_step_idx: Dict[int, int] = {
        int(raw_t): int(step_idx)
        for step_idx, raw_t in enumerate(timestep_map_reversed)
    }

    # --- Eval config ---
    total_eval_images = max(1, int(eval_num_images))
    chunk_size = max(1, min(int(eval_chunk_size), total_eval_images))
    LOGGER.info(
        "Eval config: total_images=%d chunk_size=%d (actual GPU batch=2*chunk_size=%d for CFG)",
        total_eval_images,
        chunk_size,
        2 * chunk_size,
    )

    # --- Threshold config ---
    blockwise_by_runtime: Optional[Dict[str, Dict[str, Any]]] = None
    blockwise_by_id: Optional[Dict[int, Dict[str, Any]]] = None
    threshold_mode = "global"
    threshold_meta_diag: Dict[str, Any] = {
        "threshold_mode": "global",
        "global_zone_l1": float(zone_l1_threshold),
        "global_peak_l1": float(peak_l1_threshold),
        "note": (
            "Single global thresholds from CLI. Use build_blockwise_thresholds_dit.py + "
            "--threshold-config for per-block thresholds (Pass 2)."
        ),
    }
    if threshold_config_path:
        blockwise_by_runtime, blockwise_by_id, tc_doc = _load_blockwise_threshold_config(
            threshold_config_path
        )
        threshold_mode = "blockwise_quantile"
        threshold_meta_diag = {
            "threshold_mode": threshold_mode,
            "threshold_config_path": str(Path(threshold_config_path).resolve()),
            "method": tc_doc.get("method"),
            "source": tc_doc.get("source"),
            "source_diagnostics_path": tc_doc.get("source_diagnostics_path"),
            "q_zone": tc_doc.get("q_zone"),
            "q_peak": tc_doc.get("q_peak"),
            "peak_over_zone_ratio_min": tc_doc.get("peak_over_zone_ratio_min"),
        }

    # --- Error collection loop ---
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    agg_step_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    done = 0
    chunk_idx = 0

    while done < total_eval_images:
        bsz = min(chunk_size, total_eval_images - done)
        latent_shape = (bsz, 4, latent_size, latent_size)

        # Fixed latent noise（確保 baseline 和 cache pass 使用完全相同的 noise）
        z_T = torch.randn(latent_shape, generator=rng, device=device)
        z_T_cfg = torch.cat([z_T, z_T], dim=0)  # CFG batched: (2B, 4, H, W)

        # Fixed class labels（class 0~7 循環）
        class_ids = [FIXED_EVAL_CLASSES[i % num_classes] for i in range(done, done + bsz)]
        y_cond = torch.tensor(class_ids, dtype=torch.long, device=device)
        y_null = torch.tensor([num_classes] * bsz, dtype=torch.long, device=device)
        y = torch.cat([y_cond, y_null], dim=0)  # (2B,)

        collector = Stage2ErrorCollectorDiT(T=T, batch_size_per_side=bsz, device=device)
        cb = collector.make_residual_callback()

        try:
            # Pass A: Baseline（全 full compute，cache_enabled=False）
            collector.set_run("baseline")
            _seed_all(seed + chunk_idx * 2)
            with DiTStage2Context(
                model=model,
                recompute_step_indices_by_block=cache_sched_effective,
                raw_t_to_step_idx=raw_t_to_step_idx,
                callback=cb,
                cache_enabled=False,
                T=T,
            ) as baseline_ctx:
                _run_single_sampling_pass(
                    model=model,
                    diffusion=diffusion,
                    z_T=z_T_cfg,
                    y=y,
                    cfg_scale=cfg_scale,
                    device=device,
                    sampler=sampler,
                    eta=eta,
                )
            LOGGER.info(
                "%s | baseline_stats=%s",
                collector.debug_snapshot_line(f"after_baseline_chunk_{chunk_idx}"),
                baseline_ctx.aggregate_stats(),
            )

            # Pass B: Cache mode（按 cache_sched_effective，cache_enabled=True）
            collector.set_run("cache")
            _seed_all(seed + chunk_idx * 2)
            with DiTStage2Context(
                model=model,
                recompute_step_indices_by_block=cache_sched_effective,
                raw_t_to_step_idx=raw_t_to_step_idx,
                callback=cb,
                cache_enabled=True,
                T=T,
            ) as cache_ctx:
                _run_single_sampling_pass(
                    model=model,
                    diffusion=diffusion,
                    z_T=z_T_cfg,
                    y=y,
                    cfg_scale=cfg_scale,
                    device=device,
                    sampler=sampler,
                    eta=eta,
                )
            LOGGER.info(
                "%s | cache_stats=%s",
                collector.debug_snapshot_line(f"after_cache_chunk_{chunk_idx}"),
                cache_ctx.aggregate_stats(),
            )

            chunk_diag = collector.compute_diagnostics(shared_zones)
            _aggregate_step_metrics_inplace(
                agg_step_metrics,
                chunk_diag["per_block_step_error"],
                batch_size=bsz,
            )
        finally:
            collector.clear_storage()
            del collector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        done += bsz
        chunk_idx += 1
        LOGGER.info("Eval progress: %d/%d images done.", done, total_eval_images)

    # --- Build final diagnostics ---
    per_block_step_agg = _finalize_per_block_step_error(agg_step_metrics)
    diagnostics = _build_diagnostics(
        per_block_step_error=per_block_step_agg,
        shared_zones=shared_zones,
        T=T,
    )
    diagnostics["cache_scheduler_input"] = cache_scheduler_to_jsonable(cache_sched_stage1)
    diagnostics["cache_scheduler_effective_for_cache_pass"] = cache_scheduler_to_jsonable(
        cache_sched_effective
    )
    diagnostics["cache_scheduler_runtime_overrides"] = dict(override_meta)
    diagnostics["scheduler_config_path"] = str(Path(scheduler_config_path).resolve())
    diagnostics["sampler"] = sampler
    if sampler == "ddim":
        diagnostics["eta"] = float(eta)
    diagnostics["stage2_threshold_meta"] = threshold_meta_diag
    diagnostics["eval_config"] = {
        "eval_num_images": int(total_eval_images),
        "eval_chunk_size": int(chunk_size),
        "seed": int(seed),
        "cfg_scale": float(cfg_scale),
        "sampler": sampler,
        "fixed_class_labels": [FIXED_EVAL_CLASSES[i % num_classes] for i in range(total_eval_images)],
        "model_name": str(model_name),
    }
    if sampler == "ddim":
        diagnostics["eval_config"]["eta"] = float(eta)

    per_block_step = diagnostics["per_block_step_error"]
    per_block_zone = diagnostics["per_block_zone_error"]
    per_si = aggregate_per_step_index(per_block_step)

    # --- Refinement: zone-level k adjustment ---
    refined = copy.deepcopy(cfg)
    refined["version"] = "stage2_refined_v1_dit"
    refined["stage2_meta"] = {
        "zone_l1_threshold": float(zone_l1_threshold),
        "peak_l1_threshold": float(peak_l1_threshold),
        "seed": int(seed),
        "sampler": sampler,
        "threshold_mode": threshold_mode,
        "threshold_config_path": str(Path(threshold_config_path).resolve())
        if threshold_config_path
        else None,
        "cache_runtime_overrides": {
            "force_full_prefix_steps": int(force_full_prefix_steps),
            "force_full_runtime_blocks_effective": list(blocks_eff),
        },
    }
    if sampler == "ddim":
        refined["stage2_meta"]["eta"] = float(eta)

    k_touch: List[Dict[str, Any]] = []
    blocks = sorted(refined["blocks"], key=lambda b: int(b["id"]))

    for b in blocks:
        rt = stage1_block_to_runtime_block(str(b["name"]))
        runtime_bid = runtime_name_to_block_id(rt)
        if blockwise_by_runtime is not None and rt not in blockwise_by_runtime:
            raise RuntimeError(f"threshold config missing runtime_name {rt} (block id={b['id']})")

        zone_thr_used = (
            float(blockwise_by_runtime[rt]["zone_l1_threshold"])
            if blockwise_by_runtime is not None
            else float(zone_l1_threshold)
        )

        kz = [int(x) for x in b["k_per_zone"]]
        for z in shared_zones:
            zid = int(z["id"])
            st = per_block_zone.get(rt, {}).get(str(zid), {})
            ml1 = float(st.get("mean_l1", 0.0))
            if not math.isnan(ml1) and ml1 > zone_thr_used:
                if zid < 0 or zid >= len(kz):
                    raise RuntimeError(f"block {b['id']}: bad zone id {zid}")
                old = kz[zid]
                kz[zid] = max(1, old - 1)
                k_touch.append(
                    {
                        "block_id": b["id"],
                        "scheduler_local_block_id": int(b["id"]),
                        "runtime_name": rt,
                        "canonical_runtime_block_id": int(runtime_bid),
                        "zone_id": zid,
                        "k_before": old,
                        "k_after": kz[zid],
                        "mean_l1": float(ml1),
                        "threshold_mode": threshold_mode,
                        "zone_l1_threshold_used": zone_thr_used,
                    }
                )
        b["k_per_zone"] = kz

    # 重建 expanded_mask（k_per_zone 調整後）
    for b in blocks:
        bid = int(b["id"])
        b["expanded_mask"] = rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
            shared_zones,
            [int(x) for x in b["k_per_zone"]],
            T,
            block_id=bid,
        )

    # --- Refinement: peak-level mask repair ---
    mask_touch: List[Dict[str, Any]] = []
    for b in blocks:
        rt = stage1_block_to_runtime_block(str(b["name"]))
        runtime_bid = runtime_name_to_block_id(rt)
        peak_thr_used = (
            float(blockwise_by_runtime[rt]["peak_l1_threshold"])
            if blockwise_by_runtime is not None
            else float(peak_l1_threshold)
        )
        row = list(b["expanded_mask"])

        for si_str, m in per_block_step.get(rt, {}).items():
            si = int(si_str)
            if float(m["l1"]) <= peak_thr_used:
                continue
            was_reuse = not bool(row[si])
            row[si] = True
            mask_touch.append(
                {
                    "block_id": b["id"],
                    "scheduler_local_block_id": int(b["id"]),
                    "runtime_name": rt,
                    "canonical_runtime_block_id": int(runtime_bid),
                    "step_index": si,
                    "ddpm_t": step_index_to_ddpm_t(si, T),
                    "was_reuse_before_peak_repair": was_reuse,
                    "expanded_mask_after": True,
                    "l1_error": float(m["l1"]),
                    "threshold_mode": threshold_mode,
                    "peak_l1_threshold_used": peak_thr_used,
                }
            )
        b["expanded_mask"] = row

    # 確保第一步（step_idx=0）一定是 full compute
    for b in blocks:
        rt = stage1_block_to_runtime_block(str(b["name"]))
        runtime_bid = runtime_name_to_block_id(rt)
        peak_thr_used = (
            float(blockwise_by_runtime[rt]["peak_l1_threshold"])
            if blockwise_by_runtime is not None
            else float(peak_l1_threshold)
        )
        if not bool(b["expanded_mask"][0]):
            b["expanded_mask"][0] = True
            mask_touch.append(
                {
                    "block_id": b["id"],
                    "scheduler_local_block_id": int(b["id"]),
                    "runtime_name": rt,
                    "canonical_runtime_block_id": int(runtime_bid),
                    "note": "enforce first step full compute (step_idx=0 -> DDPM t=T-1=249)",
                    "expanded_mask_after": True,
                    "threshold_mode": threshold_mode,
                    "peak_l1_threshold_used": peak_thr_used,
                }
            )

    # --- Compute full_compute_ratio ---
    total_full = sum(sum(1 for v in b["expanded_mask"] if v) for b in blocks)
    total_cells = len(blocks) * T
    full_compute_ratio = total_full / total_cells if total_cells > 0 else float("nan")

    # --- Build refined cache scheduler for diagnostics ---
    refined_cache_sched = stage1_mask_to_runtime_cache_scheduler(refined)
    diagnostics["refined_cache_scheduler"] = cache_scheduler_to_jsonable(refined_cache_sched)

    # --- Per-block threshold summary ---
    per_block_thr_summary: Optional[List[Dict[str, Any]]] = None
    if blockwise_by_id is not None:
        per_block_thr_summary = [
            {
                "block_id": int(blockwise_by_id[i]["block_id"]),
                "canonical_runtime_block_id": int(
                    blockwise_by_id[i].get("canonical_runtime_block_id", i)
                ),
                "canonical_name": blockwise_by_id[i]["canonical_name"],
                "runtime_name": blockwise_by_id[i]["runtime_name"],
                "zone_l1_threshold": float(blockwise_by_id[i]["zone_l1_threshold"]),
                "peak_l1_threshold": float(blockwise_by_id[i]["peak_l1_threshold"]),
            }
            for i in range(EXPECTED_NUM_BLOCKS)
        ]

    summary = {
        "threshold_mode": threshold_mode,
        "global_thresholds": {
            "zone_l1": float(zone_l1_threshold),
            "peak_l1": float(peak_l1_threshold),
        },
        "per_block_thresholds": per_block_thr_summary,
        "zone_k_adjustments": k_touch,
        "peak_mask_adjustments": mask_touch,
        "full_compute_ratio": float(full_compute_ratio),
        "total_full_compute_cells": int(total_full),
        "total_cells": int(total_cells),
        "zone_adj_count": len(k_touch),
        "peak_repair_count": len([m for m in mask_touch if m.get("was_reuse_before_peak_repair", False)]),
        "aggregate_per_step_index_l1": per_si,
        "eval_config": diagnostics["eval_config"],
        "cache_runtime_overrides": dict(override_meta),
    }

    # --- Write outputs ---
    def _write_json(path: Path, obj: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(obj), f, indent=2, ensure_ascii=False)

    _write_json(out_dir / "stage2_runtime_diagnostics.json", diagnostics)
    _write_json(out_dir / "stage2_refined_scheduler_config.json", refined)
    _write_json(out_dir / "stage2_refinement_summary.json", summary)
    _write_json(
        out_dir / "cache_runtime_overrides_run.json",
        {
            "stage": "stage2_runtime_refine_dit",
            "scheduler_config_path": str(Path(scheduler_config_path).resolve()),
            "T": int(T),
            "seed": int(seed),
            **dict(override_meta),
        },
    )

    LOGGER.info(
        "Stage2 refine done → %s | zone_adj=%d peak_repair=%d full_ratio=%.4f",
        out_dir,
        len(k_touch),
        summary["peak_repair_count"],
        full_compute_ratio,
    )
    print(
        f"[Stage2] {out_dir.name} | threshold_mode={threshold_mode} "
        f"zone_adj={len(k_touch)} peak_repair={summary['peak_repair_count']} "
        f"full_compute_ratio={full_compute_ratio:.4f}"
    )

    return {
        "output_dir": str(out_dir),
        "diagnostics": diagnostics,
        "summary": summary,
        "refined_config": refined,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _nonnegative_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage2 runtime refine for DiT (single-pass). "
            "Run twice: Pass1 with --threshold-mode global, "
            "Pass2 with --threshold-mode blockwise --threshold-config <path>."
        ),
    )

    g_in = p.add_argument_group("Stage1 input / output")
    g_in.add_argument("--scheduler-config", type=str, required=True,
                      help="Path to Stage1 scheduler_config.json")
    g_in.add_argument("--output-dir", type=str, required=True,
                      help="Output directory for this pass")
    g_in.add_argument("--seed", type=int, default=0)

    g_thr = p.add_argument_group("Threshold")
    g_thr.add_argument(
        "--threshold-mode",
        choices=["global", "blockwise"],
        default="global",
        help="'global' for Pass1, 'blockwise' for Pass2",
    )
    g_thr.add_argument("--zone-l1-threshold", type=float, default=DEFAULT_ZONE_L1_THRESHOLD)
    g_thr.add_argument("--peak-l1-threshold", type=float, default=DEFAULT_PEAK_L1_THRESHOLD)
    g_thr.add_argument(
        "--threshold-config",
        type=str,
        default=None,
        help="Path to stage2_thresholds_blockwise.json (required for --threshold-mode blockwise)",
    )

    g_model = p.add_argument_group("DiT model")
    g_model.add_argument("--model", type=str, default=DEFAULT_DIT_MODEL)
    g_model.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    g_model.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    g_model.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    g_model.add_argument("--num-sampling-steps", type=int, default=DEFAULT_NUM_SAMPLING_STEPS)
    g_model.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    g_model.add_argument("--eta", type=float, default=0.0, help="DDIM eta; only used with --sampler ddim.")
    g_model.add_argument("--ckpt", type=str, default=None)

    g_eval = p.add_argument_group("Diagnostics eval")
    g_eval.add_argument("--eval-num-images", type=_nonnegative_int, default=DEFAULT_EVAL_NUM_IMAGES)
    g_eval.add_argument("--eval-chunk-size", type=_nonnegative_int, default=DEFAULT_EVAL_CHUNK_SIZE)

    g_safe = p.add_argument_group("Safety overrides (cache pass only)")
    g_safe.add_argument("--force-full-prefix-steps", type=_nonnegative_int, default=0)
    g_safe.add_argument(
        "--force-full-runtime-blocks",
        type=str,
        default="",
        help="Comma-separated runtime block names (e.g. 'block_0,block_1') to force full compute.",
    )

    args = p.parse_args()

    if args.threshold_mode == "blockwise" and not args.threshold_config:
        p.error("--threshold-config is required when --threshold-mode blockwise")

    force_blocks: List[str] = []
    if args.force_full_runtime_blocks:
        for tok in args.force_full_runtime_blocks.split(","):
            tok = tok.strip()
            if tok:
                force_blocks.append(tok)

    _configure_stage2_logging()
    LOGGER.info(
        "----- Stage2-DiT run start | scheduler_config=%s | output_dir=%s | seed=%s "
        "| threshold_mode=%s -----",
        args.scheduler_config,
        args.output_dir,
        args.seed,
        args.threshold_mode,
    )

    run_stage2_refine_dit(
        scheduler_config_path=args.scheduler_config,
        output_dir=args.output_dir,
        seed=int(args.seed),
        zone_l1_threshold=float(args.zone_l1_threshold),
        peak_l1_threshold=float(args.peak_l1_threshold),
        threshold_config_path=args.threshold_config if args.threshold_mode == "blockwise" else None,
        model_name=str(args.model),
        image_size=int(args.image_size),
        num_classes=int(args.num_classes),
        cfg_scale=float(args.cfg_scale),
        num_sampling_steps=int(args.num_sampling_steps),
        sampler=str(args.sampler),
        eta=float(args.eta),
        ckpt=args.ckpt,
        eval_num_images=int(args.eval_num_images),
        eval_chunk_size=int(args.eval_chunk_size),
        force_full_prefix_steps=int(args.force_full_prefix_steps),
        force_full_runtime_blocks=force_blocks,
    )


if __name__ == "__main__":
    main()
