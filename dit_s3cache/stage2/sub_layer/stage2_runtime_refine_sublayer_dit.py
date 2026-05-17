"""Runtime Stage 2 refinement for DiT sub-layer caching.

This is the sub-layer counterpart of ``stage2_runtime_refine_dit.py``.  It
runs real DiT baseline/cache sampling passes and compares the actual gated
branch residuals ``r_msa`` / ``r_mlp``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diffusion import create_diffusion
from download import find_model
from models import DiT_models, modulate

from dit_s3cache.stage1.stage1_dit_sublayer import build_expanded_mask_for_sublayer
from dit_s3cache.stage2.sub_layer.stage2_dit_sublayer import (
    build_sublayerwise_thresholds,
    load_sublayer_scheduler_config,
    validate_sublayer_threshold_config,
)
from dit_s3cache.stage1.stage1_scheduler_dit import step_index_to_ddim_t


LOGGER = logging.getLogger("Stage2RuntimeRefineDiTSubLayer")
_STAGE2_DIR = Path(__file__).resolve().parent
_STAGE2_LOG_FILE = _STAGE2_DIR / "stage2_runtime_refine_sublayer_dit.log"
_LOG_FMT = logging.Formatter("%(asctime)s [%(levelname)s] [Stage2-Sub] %(message)s")

DEFAULT_EVAL_NUM_IMAGES = 8
DEFAULT_EVAL_CHUNK_SIZE = 1
DEFAULT_ZONE_L1_THRESHOLD = 0.02
DEFAULT_PEAK_L1_THRESHOLD = 0.08
DEFAULT_DIT_MODEL = "DiT-XL/2"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_NUM_CLASSES = 1000
DEFAULT_CFG_SCALE = 1.5
DEFAULT_NUM_SAMPLING_STEPS = 250
FIXED_EVAL_CLASSES = list(range(1000))
BRANCHES = ("msa", "mlp")


def _configure_logging() -> None:
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


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sublayer_name(block_idx: int, branch: str) -> str:
    return f"block_{block_idx:02d}_{branch}"


def _l1_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def _l2_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).mean().sqrt()


def _flatten_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1, eps=1e-8)


class Stage2SubLayerOnlineCollector:
    """Store baseline residuals and compute cache-vs-baseline metrics online."""

    def __init__(self, T: int, batch_size_per_side: int) -> None:
        self.T = int(T)
        self.B = int(batch_size_per_side)
        self._run_name = "baseline"
        self._baseline: dict[str, dict[int, torch.Tensor]] = {}
        self._step_metrics: dict[str, dict[str, dict[str, float]]] = {}
        self.baseline_writes = 0
        self.cache_compares = 0

    def set_run(self, name: str) -> None:
        if name not in ("baseline", "cache"):
            raise ValueError("run name must be 'baseline' or 'cache'")
        self._run_name = name

    def clear_storage(self) -> None:
        self._baseline.clear()
        self._step_metrics.clear()

    def make_callback(self) -> Callable[[str, torch.Tensor, int], None]:
        return self.on_sublayer_residual

    def on_sublayer_residual(self, name: str, residual: torch.Tensor, step_idx: int) -> None:
        if not (0 <= int(step_idx) < self.T):
            raise RuntimeError(f"{name}: step_idx out of range: {step_idx}")
        cond = residual[: self.B].detach().to(device="cpu", dtype=torch.float16)
        if self._run_name == "baseline":
            self._baseline.setdefault(name, {})[int(step_idx)] = cond
            self.baseline_writes += 1
            return

        base = self._baseline.get(name, {}).pop(int(step_idx), None)
        if base is None:
            raise RuntimeError(f"missing baseline residual for {name} step_idx={step_idx}")
        a = base.float()
        b = cond.float()
        l1 = float(_l1_scalar(a, b))
        l2 = float(_l2_rms(a, b))
        cos = float(_flatten_cosine(a, b).mean())
        self._step_metrics.setdefault(name, {})[str(int(step_idx))] = {
            "l1": l1,
            "l2": l2,
            "cosine": cos,
        }
        self.cache_compares += 1

    def finalize_chunk_metrics(self) -> dict[str, dict[str, dict[str, float]]]:
        missing = sum(len(v) for v in self._baseline.values())
        if missing:
            raise RuntimeError(f"baseline residuals left unmatched after cache pass: {missing}")
        return copy.deepcopy(self._step_metrics)

    def debug_snapshot_line(self, phase: str) -> str:
        return (
            f"[collector debug] phase={phase} baseline_writes={self.baseline_writes} "
            f"cache_compares={self.cache_compares} baseline_pending={sum(len(v) for v in self._baseline.values())}"
        )


class DiTStage2SubLayerBlock:
    """Patch one DiTBlock with independent MSA/MLP runtime cache and callbacks."""

    def __init__(
        self,
        block: torch.nn.Module,
        block_idx: int,
        recompute_by_branch: dict[str, set[int]],
        raw_t_to_step_idx: dict[int, int],
        callback: Callable[[str, torch.Tensor, int], None] | None,
        cache_enabled: bool,
    ) -> None:
        self.block = block
        self.block_idx = int(block_idx)
        self.recompute_by_branch = {k: set(v) for k, v in recompute_by_branch.items()}
        self.raw_t_to_step_idx = dict(raw_t_to_step_idx)
        self.callback = callback
        self.cache_enabled = bool(cache_enabled)
        self.current_raw_t: int | None = None
        self.cached_msa: torch.Tensor | None = None
        self.cached_mlp: torch.Tensor | None = None
        self.msa_cache_hits = 0
        self.msa_recompute_hits = 0
        self.mlp_cache_hits = 0
        self.mlp_recompute_hits = 0
        self._original_forward = block.forward
        self._is_installed = False
        self.install()

    def install(self) -> None:
        if self._is_installed:
            return
        block = self.block

        def hooked_forward(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            raw_t = self.current_raw_t
            if raw_t is None:
                raise RuntimeError(f"block_{self.block_idx:02d}: current_raw_t not set")
            step_idx = self.raw_t_to_step_idx.get(int(raw_t))
            if step_idx is None:
                raise RuntimeError(f"raw_t={raw_t} not in raw_t_to_step_idx")

            compute_msa = (
                not self.cache_enabled
                or step_idx in self.recompute_by_branch.get("msa", set())
                or self.cached_msa is None
            )
            compute_mlp = (
                not self.cache_enabled
                or step_idx in self.recompute_by_branch.get("mlp", set())
                or self.cached_mlp is None
            )

            if compute_msa or compute_mlp:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    block.adaLN_modulation(c).chunk(6, dim=1)
                )
            else:
                shift_msa = scale_msa = gate_msa = None
                shift_mlp = scale_mlp = gate_mlp = None

            if compute_msa:
                assert shift_msa is not None and scale_msa is not None and gate_msa is not None
                r_msa = gate_msa.unsqueeze(1) * block.attn(
                    modulate(block.norm1(x), shift_msa, scale_msa)
                )
                if self.cache_enabled:
                    self.cached_msa = r_msa.detach()
                self.msa_recompute_hits += 1
            else:
                r_msa = self.cached_msa
                self.msa_cache_hits += 1
            if r_msa is None:
                raise RuntimeError(f"block_{self.block_idx:02d}_msa cache unexpectedly empty")
            if self.callback is not None:
                self.callback(_sublayer_name(self.block_idx, "msa"), r_msa.detach(), step_idx)

            x_after_msa = x + r_msa

            if compute_mlp:
                assert shift_mlp is not None and scale_mlp is not None and gate_mlp is not None
                r_mlp = gate_mlp.unsqueeze(1) * block.mlp(
                    modulate(block.norm2(x_after_msa), shift_mlp, scale_mlp)
                )
                if self.cache_enabled:
                    self.cached_mlp = r_mlp.detach()
                self.mlp_recompute_hits += 1
            else:
                r_mlp = self.cached_mlp
                self.mlp_cache_hits += 1
            if r_mlp is None:
                raise RuntimeError(f"block_{self.block_idx:02d}_mlp cache unexpectedly empty")
            if self.callback is not None:
                self.callback(_sublayer_name(self.block_idx, "mlp"), r_mlp.detach(), step_idx)

            return x_after_msa + r_mlp

        self.block.forward = hooked_forward
        self._is_installed = True

    def restore(self) -> None:
        if not self._is_installed:
            return
        self.block.forward = self._original_forward
        self._is_installed = False
        self.cached_msa = None
        self.cached_mlp = None
        self.current_raw_t = None

    def stats(self) -> dict[str, int]:
        return {
            "msa_cache_hits": int(self.msa_cache_hits),
            "msa_recompute_hits": int(self.msa_recompute_hits),
            "mlp_cache_hits": int(self.mlp_cache_hits),
            "mlp_recompute_hits": int(self.mlp_recompute_hits),
        }


class DiTStage2SubLayerContext:
    def __init__(
        self,
        model: torch.nn.Module,
        recompute_step_indices_by_sublayer: dict[tuple[int, str], set[int]],
        raw_t_to_step_idx: dict[int, int],
        callback: Callable[[str, torch.Tensor, int], None] | None,
        cache_enabled: bool,
    ) -> None:
        self.model = model
        self.recompute = {k: set(v) for k, v in recompute_step_indices_by_sublayer.items()}
        self.raw_t_to_step_idx = dict(raw_t_to_step_idx)
        self.callback = callback
        self.cache_enabled = bool(cache_enabled)
        self._blocks: list[DiTStage2SubLayerBlock] = []
        self._original_forward_with_cfg: Callable | None = None

    def __enter__(self) -> "DiTStage2SubLayerContext":
        for block_idx, block in enumerate(self.model.blocks):
            recompute_by_branch = {
                "msa": self.recompute.get((block_idx, "msa"), set()),
                "mlp": self.recompute.get((block_idx, "mlp"), set()),
            }
            self._blocks.append(
                DiTStage2SubLayerBlock(
                    block=block,
                    block_idx=block_idx,
                    recompute_by_branch=recompute_by_branch,
                    raw_t_to_step_idx=self.raw_t_to_step_idx,
                    callback=self.callback,
                    cache_enabled=self.cache_enabled,
                )
            )

        original_fwcfg = self.model.forward_with_cfg
        stage2_blocks = self._blocks

        def wrapped_forward_with_cfg(
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
        self.model.forward_with_cfg = wrapped_forward_with_cfg
        LOGGER.info("DiTStage2SubLayerContext enabled | cache_enabled=%s", self.cache_enabled)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        for blk in self._blocks:
            blk.restore()
        self._blocks.clear()
        if self._original_forward_with_cfg is not None:
            self.model.forward_with_cfg = self._original_forward_with_cfg
            self._original_forward_with_cfg = None

    def aggregate_stats(self) -> dict[str, Any]:
        return {"per_block": {f"block_{b.block_idx:02d}": b.stats() for b in self._blocks}}


def _run_single_sampling_pass(
    *,
    model: torch.nn.Module,
    diffusion: Any,
    z_T: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float,
    device: torch.device,
) -> None:
    model_kwargs = {"y": y, "cfg_scale": cfg_scale}
    sample_iter = diffusion.p_sample_loop_progressive(
        model.forward_with_cfg,
        z_T.shape,
        noise=z_T,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
    )
    for _out in sample_iter:
        pass


def sublayer_mask_to_recompute_step_indices(cfg: dict[str, Any]) -> dict[tuple[int, str], set[int]]:
    out: dict[tuple[int, str], set[int]] = {}
    T = int(cfg["T"])
    for name, entry in cfg["sublayers"].items():
        block_idx = int(entry["block_idx"])
        branch = str(entry["branch"])
        row = entry["expanded_mask"]
        if len(row) != T:
            raise ValueError(f"{name}: expanded_mask length {len(row)} != T={T}")
        out[(block_idx, branch)] = {i for i, v in enumerate(row) if bool(v)}
    return out


def _aggregate_step_metrics_inplace(
    agg: dict[str, dict[str, dict[str, float]]],
    per_step_error: dict[str, dict[str, dict[str, float]]],
    *,
    batch_size: int,
) -> None:
    w = float(batch_size)
    for name, steps in per_step_error.items():
        row = agg.setdefault(name, {})
        for si_str, m in steps.items():
            slot = row.setdefault(si_str, {"sum_l1": 0.0, "sum_l2_sq": 0.0, "sum_cos": 0.0, "weight": 0.0})
            l1 = float(m["l1"])
            l2 = float(m["l2"])
            cos = float(m["cosine"])
            slot["sum_l1"] += l1 * w
            slot["sum_l2_sq"] += (l2 * l2) * w
            slot["sum_cos"] += cos * w
            slot["weight"] += w


def _finalize_step_error(agg: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, steps in agg.items():
        row: dict[str, dict[str, float]] = {}
        for si_str, slot in steps.items():
            w = float(slot["weight"])
            if w <= 0:
                continue
            row[si_str] = {
                "l1": float(slot["sum_l1"] / w),
                "l2": float(math.sqrt(max(slot["sum_l2_sq"] / w, 0.0))),
                "cosine": float(slot["sum_cos"] / w),
            }
        out[name] = row
    return out


def _build_diagnostics(
    *,
    cfg: dict[str, Any],
    per_step_error: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    T = int(cfg["T"])
    per_zone: dict[str, dict[str, dict[str, Any]]] = {}
    all_l1: list[float] = []
    all_l2: list[float] = []
    all_cos: list[float] = []
    for name, entry in sorted(cfg["sublayers"].items(), key=lambda kv: int(kv[1]["flat_idx"])):
        per_zone[name] = {}
        steps = per_step_error.get(name, {})
        for m in steps.values():
            all_l1.append(float(m["l1"]))
            all_l2.append(float(m["l2"]))
            all_cos.append(float(m["cosine"]))
        for z in entry["zones"]:
            zid = int(z["zone_id"])
            step_indices = list(range(int(z["start_step"]), int(z["end_step"]) + 1))
            vals_l1 = [float(steps[str(si)]["l1"]) for si in step_indices if str(si) in steps]
            vals_l2 = [float(steps[str(si)]["l2"]) for si in step_indices if str(si) in steps]
            vals_cos = [float(steps[str(si)]["cosine"]) for si in step_indices if str(si) in steps]
            per_zone[name][str(zid)] = {
                "mean_l1": float(np.mean(vals_l1)) if vals_l1 else float("nan"),
                "mean_l2": float(np.mean(vals_l2)) if vals_l2 else float("nan"),
                "mean_cosine": float(np.mean(vals_cos)) if vals_cos else float("nan"),
                "num_steps": int(len(step_indices)),
                "num_compared_in_zone": int(len(vals_l1)),
            }

    return {
        "per_sublayer_step_error": per_step_error,
        "per_sublayer_zone_error": per_zone,
        "global_summary": {
            "mean_l1": float(np.mean(all_l1)) if all_l1 else None,
            "mean_l2": float(np.mean(all_l2)) if all_l2 else None,
            "mean_cosine": float(np.mean(all_cos)) if all_cos else None,
            "num_entries": int(len(all_l1)),
            "cfg_half_note": "Metrics computed on cond half ([:B]) of gated r_msa/r_mlp branch residuals.",
        },
        "T": T,
        "granularity": "sublayer",
        "error_source": "runtime baseline/cache sampling residual comparison",
    }


def _mask_from_config(cfg: dict[str, Any]) -> np.ndarray:
    T = int(cfg["T"])
    mask = np.zeros((T, len(cfg["sublayers"])), dtype=bool)
    for _, entry in cfg["sublayers"].items():
        mask[:, int(entry["flat_idx"])] = np.asarray(entry["expanded_mask"], dtype=bool)
    return mask


def _load_sublayer_threshold_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_sublayer_threshold_config(data)
    return data


def run_stage2_runtime_refine_sublayer_dit(
    *,
    scheduler_config_path: str,
    output_dir: str,
    seed: int = 42,
    zone_l1_threshold: float = DEFAULT_ZONE_L1_THRESHOLD,
    peak_l1_threshold: float = DEFAULT_PEAK_L1_THRESHOLD,
    threshold_config_path: str | None = None,
    model_name: str = DEFAULT_DIT_MODEL,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    num_sampling_steps: int = DEFAULT_NUM_SAMPLING_STEPS,
    ckpt: str | None = None,
    eval_num_images: int = DEFAULT_EVAL_NUM_IMAGES,
    eval_chunk_size: int = DEFAULT_EVAL_CHUNK_SIZE,
) -> dict[str, Any]:
    _configure_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runtime sub-layer Stage2 refine.")

    device = torch.device("cuda")
    _seed_all(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_sublayer_scheduler_config(scheduler_config_path)
    T = int(cfg["T"])
    cache_sched = sublayer_mask_to_recompute_step_indices(cfg)

    latent_size = image_size // 8
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    LOGGER.info("Loading DiT model: %s", model_name)
    model = DiT_models[model_name](input_size=latent_size, num_classes=num_classes).to(device)
    ckpt_path = ckpt or f"DiT-XL-2-{image_size}x{image_size}.pt"
    model.load_state_dict(find_model(ckpt_path))
    model.eval()
    diffusion = create_diffusion(str(num_sampling_steps))
    timestep_map_reversed = list(reversed(diffusion.timestep_map))[:T]
    if len(timestep_map_reversed) != T:
        raise RuntimeError(f"Unexpected timestep map length {len(timestep_map_reversed)} != T={T}")
    raw_t_to_step_idx = {int(raw_t): int(step_idx) for step_idx, raw_t in enumerate(timestep_map_reversed)}

    threshold_doc = None
    thresholds = None
    threshold_mode = "sublayerwise_quantile" if threshold_config_path else "global"
    if threshold_config_path:
        threshold_doc = _load_sublayer_threshold_config(threshold_config_path)
        thresholds = threshold_doc["thresholds"]

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    agg_step_metrics: dict[str, dict[str, dict[str, float]]] = {}
    total_eval_images = max(1, int(eval_num_images))
    chunk_size = max(1, min(int(eval_chunk_size), total_eval_images))

    done = 0
    chunk_idx = 0
    while done < total_eval_images:
        bsz = min(chunk_size, total_eval_images - done)
        z_T = torch.randn((bsz, 4, latent_size, latent_size), generator=rng, device=device)
        z_T_cfg = torch.cat([z_T, z_T], dim=0)
        class_ids = [FIXED_EVAL_CLASSES[i % num_classes] for i in range(done, done + bsz)]
        y_cond = torch.tensor(class_ids, dtype=torch.long, device=device)
        y_null = torch.tensor([num_classes] * bsz, dtype=torch.long, device=device)
        y = torch.cat([y_cond, y_null], dim=0)

        collector = Stage2SubLayerOnlineCollector(T=T, batch_size_per_side=bsz)
        cb = collector.make_callback()
        try:
            collector.set_run("baseline")
            _seed_all(seed + chunk_idx * 2)
            with DiTStage2SubLayerContext(
                model=model,
                recompute_step_indices_by_sublayer=cache_sched,
                raw_t_to_step_idx=raw_t_to_step_idx,
                callback=cb,
                cache_enabled=False,
            ) as baseline_ctx:
                _run_single_sampling_pass(
                    model=model,
                    diffusion=diffusion,
                    z_T=z_T_cfg,
                    y=y,
                    cfg_scale=cfg_scale,
                    device=device,
                )
            LOGGER.info("%s | baseline_stats=%s", collector.debug_snapshot_line("baseline"), baseline_ctx.aggregate_stats())

            collector.set_run("cache")
            _seed_all(seed + chunk_idx * 2)
            with DiTStage2SubLayerContext(
                model=model,
                recompute_step_indices_by_sublayer=cache_sched,
                raw_t_to_step_idx=raw_t_to_step_idx,
                callback=cb,
                cache_enabled=True,
            ) as cache_ctx:
                _run_single_sampling_pass(
                    model=model,
                    diffusion=diffusion,
                    z_T=z_T_cfg,
                    y=y,
                    cfg_scale=cfg_scale,
                    device=device,
                )
            LOGGER.info("%s | cache_stats=%s", collector.debug_snapshot_line("cache"), cache_ctx.aggregate_stats())
            _aggregate_step_metrics_inplace(
                agg_step_metrics,
                collector.finalize_chunk_metrics(),
                batch_size=bsz,
            )
        finally:
            collector.clear_storage()
            del collector
            torch.cuda.empty_cache()
        done += bsz
        chunk_idx += 1
        LOGGER.info("Eval progress: %d/%d images done.", done, total_eval_images)

    per_step = _finalize_step_error(agg_step_metrics)
    diagnostics = _build_diagnostics(cfg=cfg, per_step_error=per_step)
    diagnostics["scheduler_config_path"] = str(Path(scheduler_config_path).resolve())
    diagnostics["stage2_threshold_meta"] = {
        "threshold_mode": threshold_mode,
        "global_zone_l1": float(zone_l1_threshold),
        "global_peak_l1": float(peak_l1_threshold),
        "threshold_config_path": str(Path(threshold_config_path).resolve()) if threshold_config_path else None,
        "q_zone": threshold_doc.get("q_zone") if threshold_doc else None,
        "q_peak": threshold_doc.get("q_peak") if threshold_doc else None,
        "peak_over_zone_ratio_min": threshold_doc.get("peak_over_zone_ratio_min") if threshold_doc else None,
    }
    diagnostics["eval_config"] = {
        "eval_num_images": int(total_eval_images),
        "eval_chunk_size": int(chunk_size),
        "seed": int(seed),
        "cfg_scale": float(cfg_scale),
        "model_name": model_name,
    }

    refined = copy.deepcopy(cfg)
    refined["version"] = "stage2_runtime_sublayer_refined_v1"
    refined["stage2_meta"] = diagnostics["stage2_threshold_meta"]
    original_ratio = float(_mask_from_config(cfg).mean())
    zone_adjustments: list[dict[str, Any]] = []
    peak_repairs: list[dict[str, Any]] = []
    per_sublayer_summary: dict[str, Any] = {}

    for name, entry in sorted(refined["sublayers"].items(), key=lambda kv: int(kv[1]["flat_idx"])):
        zone_thr = float(thresholds[name]["zone_l1_threshold"]) if thresholds is not None else float(zone_l1_threshold)
        peak_thr = float(thresholds[name]["peak_l1_threshold"]) if thresholds is not None else float(peak_l1_threshold)
        zones = entry["zones"]
        local_zone_adj = 0
        for z in zones:
            zid = int(z["zone_id"])
            mean_l1 = float(diagnostics["per_sublayer_zone_error"][name][str(zid)]["mean_l1"])
            if math.isfinite(mean_l1) and mean_l1 > zone_thr:
                old_k = int(z["k"])
                new_k = max(1, old_k - 1)
                if new_k != old_k:
                    z["k"] = int(new_k)
                    local_zone_adj += 1
                    zone_adjustments.append(
                        {
                            "sublayer": name,
                            "flat_idx": int(entry["flat_idx"]),
                            "zone_id": zid,
                            "k_before": old_k,
                            "k_after": new_k,
                            "mean_l1": mean_l1,
                            "zone_l1_threshold_used": zone_thr,
                            "threshold_mode": threshold_mode,
                        }
                    )

        # Match the block-level Stage2 contract: rebuild from the current
        # zones/k state, then apply peak repairs for this pass. Pass 1 is only
        # used to derive diagnostics/thresholds; Pass 2 starts from Stage1.
        row = build_expanded_mask_for_sublayer(zones, T)
        local_peak = 0
        for si_str, metric in diagnostics["per_sublayer_step_error"][name].items():
            si = int(si_str)
            if float(metric["l1"]) > peak_thr and not bool(row[si]):
                row[si] = True
                local_peak += 1
                peak_repairs.append(
                    {
                        "sublayer": name,
                        "flat_idx": int(entry["flat_idx"]),
                        "step_index": si,
                        "ddpm_t": step_index_to_ddim_t(si, T),
                        "was_reuse_before_peak_repair": True,
                        "l1_error": float(metric["l1"]),
                        "peak_l1_threshold_used": peak_thr,
                        "threshold_mode": threshold_mode,
                    }
                )
        row[0] = True
        entry["expanded_mask"] = row.tolist()
        per_sublayer_summary[name] = {
            "peak_repairs": local_peak,
            "zone_adjustments": local_zone_adj,
            "sublayer_recompute_ratio": float(row.mean()),
            "zone_l1_threshold_used": zone_thr,
            "peak_l1_threshold_used": peak_thr,
        }

    refined_ratio = float(_mask_from_config(refined).mean())
    refined["full_compute_ratio"] = refined_ratio
    summary = {
        "threshold_mode": threshold_mode,
        "source_scheduler": Path(scheduler_config_path).parent.name,
        "original_full_compute_ratio": original_ratio,
        "refined_full_compute_ratio": refined_ratio,
        "total_peak_repairs": len(peak_repairs),
        "total_zone_k_adjustments": len(zone_adjustments),
        "zone_k_adjustments": zone_adjustments,
        "peak_mask_adjustments": peak_repairs,
        "per_sublayer_summary": per_sublayer_summary,
        "eval_config": diagnostics["eval_config"],
    }
    diagnostics["refined_recompute_summary"] = {
        "original_full_compute_ratio": original_ratio,
        "refined_full_compute_ratio": refined_ratio,
        "total_peak_repairs": len(peak_repairs),
        "total_zone_k_adjustments": len(zone_adjustments),
    }

    _write_json(out_dir / "stage2_runtime_diagnostics.json", diagnostics)
    _write_json(out_dir / "stage2_refined_scheduler_config.json", refined)
    _write_json(out_dir / "stage2_refinement_summary.json", summary)
    _write_json(
        out_dir / "cache_runtime_overrides_run.json",
        {
            "stage": "stage2_runtime_refine_sublayer_dit",
            "scheduler_config_path": str(Path(scheduler_config_path).resolve()),
            "T": T,
            "seed": seed,
        },
    )
    print(
        f"[Stage2-SubRuntime] {out_dir.name} | threshold_mode={threshold_mode} "
        f"zone_adj={len(zone_adjustments)} peak_repair={len(peak_repairs)} "
        f"full_compute_ratio={refined_ratio:.4f}"
    )
    return {"diagnostics": diagnostics, "summary": summary, "refined_config": refined}


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, ensure_ascii=False)


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
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.floating):
        return _json_safe(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description="Runtime Stage2 refine for DiT sub-layer cache schedulers")
    p.add_argument("--scheduler-config", default=None)
    p.add_argument("--threshold-mode", choices=["global", "sublayerwise"], default="global")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--zone-l1-threshold", type=float, default=DEFAULT_ZONE_L1_THRESHOLD)
    p.add_argument("--peak-l1-threshold", type=float, default=DEFAULT_PEAK_L1_THRESHOLD)
    p.add_argument("--threshold-config", default=None)
    p.add_argument("--model", default=DEFAULT_DIT_MODEL)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    p.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    p.add_argument("--num-sampling-steps", type=int, default=DEFAULT_NUM_SAMPLING_STEPS)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--eval-num-images", type=int, default=DEFAULT_EVAL_NUM_IMAGES)
    p.add_argument("--eval-chunk-size", type=int, default=DEFAULT_EVAL_CHUNK_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--build-thresholds", action="store_true")
    p.add_argument("--diagnostics", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--q-zone", type=float, default=0.90)
    p.add_argument("--q-peak", type=float, default=0.80)
    p.add_argument("--peak-over-zone-ratio-min", type=float, default=1.3)
    args = p.parse_args()

    if args.build_thresholds:
        if not args.diagnostics or not args.output:
            p.error("--build-thresholds requires --diagnostics and --output")
        build_sublayerwise_thresholds(
            diagnostics_path=args.diagnostics,
            output_path=args.output,
            q_zone=args.q_zone,
            q_peak=args.q_peak,
            peak_over_zone_ratio_min=args.peak_over_zone_ratio_min,
        )
        print(f"Done thresholds: {args.output}")
        return

    if not args.scheduler_config or not args.output_dir:
        p.error("refine mode requires --scheduler-config and --output-dir")
    if args.threshold_mode == "sublayerwise" and not args.threshold_config:
        p.error("--threshold-config is required for --threshold-mode sublayerwise")

    run_stage2_runtime_refine_sublayer_dit(
        scheduler_config_path=args.scheduler_config,
        output_dir=args.output_dir,
        seed=args.seed,
        zone_l1_threshold=args.zone_l1_threshold,
        peak_l1_threshold=args.peak_l1_threshold,
        threshold_config_path=args.threshold_config if args.threshold_mode == "sublayerwise" else None,
        model_name=args.model,
        image_size=args.image_size,
        num_classes=args.num_classes,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        ckpt=args.ckpt,
        eval_num_images=args.eval_num_images,
        eval_chunk_size=args.eval_chunk_size,
    )


if __name__ == "__main__":
    main()
