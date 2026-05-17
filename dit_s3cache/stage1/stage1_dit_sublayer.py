"""Stage 1 scheduler synthesis for DiT sub-layer caching.

This is a parallel sub-layer path.  It reads ``stage0_output_sublayer`` and
builds independent zones for each of the 56 cacheable units:

    block_00_msa, block_00_mlp, ..., block_27_mlp

The output mask layout is ``(T, 56)`` with ``True`` meaning COMPUTE.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dit_s3cache.evidence.hooks_sublayer import BRANCHES, sublayer_name
from dit_s3cache.stage1.stage1_scheduler_dit import (
    ddim_t_to_step_index,
    delta_adjacent,
    expand_zone_mask_ddim,
    interval_j_to_reused_ddim_t,
    merge_short_zones_step,
    moving_average,
    processing_order_series,
    step_index_to_ddim_t,
    topk_change_point_indices,
    zones_from_step_boundaries,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Stage1Sub] %(message)s")
LOGGER = logging.getLogger("Stage1DiTSubLayer")

VERSION = "stage1_sublayer_v1"
DEFAULT_K_SWEEP = (10, 15, 20, 25)
DEFAULT_SWEEP_WINDOWS = (2, 3, 5)
DEFAULT_LAMBDA_SWEEP = (0.5, 1.0)


def flat_to_block_branch(flat_idx: int) -> tuple[int, str]:
    return int(flat_idx) // 2, "msa" if int(flat_idx) % 2 == 0 else "mlp"


def parse_sublayer_name(name: str) -> tuple[int, str]:
    parts = str(name).split("_")
    if len(parts) != 3 or parts[0] != "block" or parts[2] not in BRANCHES:
        raise ValueError(f"invalid sub-layer name: {name!r}")
    return int(parts[1]), parts[2]


def load_stage0_sublayer(
    input_dir: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
    p = Path(input_dir)
    required = [
        "sub_layer_names.npy",
        "l1_interval_norm.npy",
        "cosdist_interval_norm.npy",
        "svd_interval_norm.npy",
        "fid_w_clip.npy",
        "axis_interval_def.npy",
        "t_curr_interval.npy",
    ]
    for fname in required:
        if not (p / fname).exists():
            raise FileNotFoundError(f"missing Stage 0 sub-layer file: {p / fname}")

    names = [str(x) for x in np.load(p / "sub_layer_names.npy", allow_pickle=True).tolist()]
    l1 = np.load(p / "l1_interval_norm.npy").astype(np.float64)
    cos = np.load(p / "cosdist_interval_norm.npy").astype(np.float64)
    svd = np.load(p / "svd_interval_norm.npy").astype(np.float64)
    fid_w = np.load(p / "fid_w_clip.npy").astype(np.float64)
    axis_def_raw = np.load(p / "axis_interval_def.npy", allow_pickle=True)
    t_curr = np.load(p / "t_curr_interval.npy").astype(np.int32).reshape(-1)

    S = len(names)
    Tm1 = l1.shape[1]
    if l1.shape != (S, Tm1) or cos.shape != (S, Tm1) or svd.shape != (S, Tm1):
        raise ValueError(f"bad evidence shapes: names={S}, l1={l1.shape}, cos={cos.shape}, svd={svd.shape}")
    if fid_w.shape != (S,):
        raise ValueError(f"fid_w_clip shape {fid_w.shape} != ({S},)")

    T = Tm1 + 1
    expected_t_curr = np.arange(T - 2, -1, -1, dtype=np.int32)
    if not np.array_equal(t_curr, expected_t_curr):
        raise ValueError(
            f"t_curr_interval mismatch: got head={t_curr[:6].tolist()}, "
            f"expected head={expected_t_curr[:6].tolist()}"
        )

    expected_names = [sublayer_name(b, branch) for b in range(S // 2) for branch in BRANCHES]
    if names != expected_names:
        raise ValueError(f"unexpected sub_layer_names order: got {names[:4]}, expected {expected_names[:4]}")

    axis_def = str(axis_def_raw) if np.ndim(axis_def_raw) == 0 else str(axis_def_raw.item())
    return names, _clip01(l1), _clip01(cos), _clip01(svd), _clip01(fid_w), axis_def, t_curr


def build_I_cut_per_ddpm_t(
    l1: np.ndarray,
    cos: np.ndarray,
    svd: np.ndarray,
    T: int,
    lambda_l1: float,
) -> np.ndarray:
    """Return ``I_cut[s, ddpm_t]`` with ``I_cut[:, T-1] = 0``."""

    if not 0.0 <= lambda_l1 <= 1.0:
        raise ValueError(f"lambda_l1 must be in [0,1], got {lambda_l1}")
    S = l1.shape[0]
    I_cut = np.zeros((S, T), dtype=np.float64)
    for j in range(T - 1):
        t = interval_j_to_reused_ddim_t(j, T)
        I_cut[:, t] = lambda_l1 * l1[:, j] + (1.0 - lambda_l1) * cos[:, j] + svd[:, j]
    I_cut[:, T - 1] = 0.0
    return I_cut


def detect_sublayer_zones(
    I_cut_s: np.ndarray,
    T: int,
    K: int,
    smooth_window: int,
    min_zone_len: int,
) -> tuple[list[dict[str, Any]], np.ndarray, list[int]]:
    proc = processing_order_series(I_cut_s, T)
    smooth = moving_average(proc, smooth_window)
    delta = delta_adjacent(smooth)
    K_eff = min(int(K), max(0, T - 1))
    cps = topk_change_point_indices(delta, K_eff, 1, T - 1)
    boundaries = sorted(set([0] + cps + [T]))
    step_zones = merge_short_zones_step(zones_from_step_boundaries(boundaries), T, min_len=min_zone_len)

    zones: list[dict[str, Any]] = []
    for zid, (s0, s1) in enumerate(step_zones):
        t_start = step_index_to_ddim_t(s0, T)
        t_end = step_index_to_ddim_t(s1, T)
        vals = [float(I_cut_s[t]) for t in range(t_end, t_start + 1)]
        zones.append(
            {
                "zone_id": int(zid),
                "start_t": int(t_start),
                "end_t": int(t_end),
                "start_step": int(s0),
                "end_step": int(s1),
                "length": int(s1 - s0 + 1),
                "avg_importance": float(np.mean(vals)) if vals else 0.0,
            }
        )
    return zones, smooth, cps


def assign_k_for_zone(
    avg_importance: float,
    fid_weight: float,
    *,
    k_min: int,
    k_max: int,
) -> int:
    """Map zone risk to k.  Higher risk -> smaller k -> more compute."""

    risk = float(np.clip(avg_importance, 0.0, None)) * float(np.clip(fid_weight, 0.0, 1.0))
    if k_max <= k_min:
        return int(k_min)
    # Importance can exceed 1 because SVD is added to l1/cos.  A 0..2 risk range
    # gives a conservative but smooth mapping across k_min..k_max.
    normalized = float(np.clip(risk / 2.0, 0.0, 1.0))
    k_float = k_max - normalized * (k_max - k_min)
    return int(np.clip(round(k_float), k_min, k_max))


def build_expanded_mask_for_sublayer(zones: list[dict[str, Any]], T: int) -> np.ndarray:
    row = np.zeros(T, dtype=bool)
    touched = np.zeros(T, dtype=bool)
    for z in zones:
        ms, _, _ = expand_zone_mask_ddim(int(z["start_t"]), int(z["end_t"]), int(z["k"]), T)
        if np.any(touched & ms):
            raise ValueError(f"overlap in expanded mask for zone {z['zone_id']}")
        row |= ms
        for step_idx in range(int(z["start_step"]), int(z["end_step"]) + 1):
            touched[step_idx] = True
    if not touched.all():
        missing = np.where(~touched)[0].tolist()
        raise ValueError(f"zones do not cover all steps; missing={missing[:16]}")
    row[0] = True
    return row


def run_stage1_sublayer(
    *,
    stage0_dir: str,
    output_dir: str,
    K: int = 15,
    smooth_window: int = 3,
    lambda_l1: float = 1.0,
    k_min: int = 1,
    k_max: int = 4,
    min_zone_len: int = 2,
) -> dict[str, Any]:
    names, l1, cos, svd, fid_w, axis_def, t_curr = load_stage0_sublayer(stage0_dir)
    S = len(names)
    T = l1.shape[1] + 1
    I_cut = build_I_cut_per_ddpm_t(l1, cos, svd, T, lambda_l1=lambda_l1)

    expanded = np.zeros((T, S), dtype=bool)
    smoothed = np.zeros((S, T), dtype=np.float64)
    sublayers: dict[str, Any] = {}
    diagnostics_sublayers: dict[str, Any] = {}

    for sidx, name in enumerate(names):
        block_idx, branch = parse_sublayer_name(name)
        zones, smooth, cps = detect_sublayer_zones(
            I_cut[sidx],
            T,
            K=K,
            smooth_window=smooth_window,
            min_zone_len=min_zone_len,
        )
        for z in zones:
            z["k"] = assign_k_for_zone(
                float(z["avg_importance"]),
                float(fid_w[sidx]),
                k_min=k_min,
                k_max=k_max,
            )

        row = build_expanded_mask_for_sublayer(zones, T)
        expanded[:, sidx] = row
        smoothed[sidx] = smooth
        sublayers[name] = {
            "flat_idx": int(sidx),
            "block_idx": int(block_idx),
            "branch": branch,
            "fid_weight": float(fid_w[sidx]),
            "zones": zones,
            "expanded_mask": row.tolist(),
        }
        diagnostics_sublayers[name] = {
            "change_points_step_index": [int(x) for x in cps],
            "num_compute": int(row.sum()),
            "num_cache": int(T - row.sum()),
            "compute_ratio": float(row.mean()),
        }

    full_compute_ratio = float(expanded.mean())
    config = {
        "version": VERSION,
        "T": int(T),
        "time_order": "ddpm_249_to_0",
        "granularity": "sublayer",
        "expanded_mask_layout": "(T, 56); row step_idx=0 is first DDPM step t=249; True=COMPUTE",
        "metadata": {
            "model": "DiT-XL/2",
            "resolution": 256,
            "T": int(T),
            "n_sublayers": int(S),
            "granularity": "sublayer",
            "K": int(K),
            "sw": int(smooth_window),
            "lambda": float(lambda_l1),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "min_zone_len": int(min_zone_len),
            "stage0_dir": str(Path(stage0_dir).resolve()),
        },
        "sublayers": sublayers,
        "full_compute_ratio": full_compute_ratio,
    }
    summary = {
        "version": VERSION,
        "stage0_dir": str(Path(stage0_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "K": int(K),
        "sw": int(smooth_window),
        "lambda": float(lambda_l1),
        "T": int(T),
        "n_sublayers": int(S),
        "full_compute_ratio": full_compute_ratio,
        "total_compute_cells": int(expanded.sum()),
        "total_cells": int(expanded.size),
        "per_sublayer_ratio": {
            name: float(expanded[:, sidx].mean()) for sidx, name in enumerate(names)
        },
        "msa_mean_compute_ratio": float(np.mean([expanded[:, i].mean() for i in range(0, S, 2)])),
        "mlp_mean_compute_ratio": float(np.mean([expanded[:, i].mean() for i in range(1, S, 2)])),
    }
    diagnostics = {
        "stage0_axis_interval_def": axis_def,
        "t_curr_interval": t_curr.tolist(),
        "I_cut_stats": _stats_dict(I_cut),
        "I_cut_formula": "lambda*l1_interval_norm + (1-lambda)*cosdist_interval_norm + svd_interval_norm",
        "mapping_note": "interval j -> reused DDPM t=(T-2)-j; expanded_mask[step_idx] maps to t=(T-1)-step_idx",
        "sublayers": diagnostics_sublayers,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scheduler_config.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(config), f, indent=2, ensure_ascii=False)
    with open(out / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
    with open(out / "scheduler_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(diagnostics), f, indent=2, ensure_ascii=False)
    np.save(out / "expanded_mask.npy", expanded.astype(bool))
    np.save(out / "I_cut_smoothed.npy", smoothed)
    LOGGER.info("wrote Stage 1 sub-layer scheduler: %s", out)
    return {"config": config, "summary": summary, "diagnostics": diagnostics}


def run_sweep(
    *,
    stage0_dir: str,
    output_root: str,
    K_values: Sequence[int],
    smooth_windows: Sequence[int],
    lambda_values: Sequence[float],
    k_min: int,
    k_max: int,
    min_zone_len: int,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for K in K_values:
        for sw in smooth_windows:
            for lam in lambda_values:
                run_name = f"sweep_K{int(K)}_sw{int(sw)}_lam{float(lam):g}"
                out_dir = Path(output_root) / run_name
                result = run_stage1_sublayer(
                    stage0_dir=stage0_dir,
                    output_dir=str(out_dir),
                    K=int(K),
                    smooth_window=int(sw),
                    lambda_l1=float(lam),
                    k_min=k_min,
                    k_max=k_max,
                    min_zone_len=min_zone_len,
                )
                reports.append(
                    {
                        "name": run_name,
                        "path": str(out_dir),
                        "K": int(K),
                        "sw": int(sw),
                        "lambda": float(lam),
                        "full_compute_ratio": result["summary"]["full_compute_ratio"],
                    }
                )
    reports_sorted = sorted(reports, key=lambda x: float(x["full_compute_ratio"]))
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "stage1_sublayer_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(reports_sorted), f, indent=2, ensure_ascii=False)
    return reports_sorted


def rebuild_expanded_mask_from_config(config: dict[str, Any]) -> np.ndarray:
    T = int(config["T"])
    items = sorted(config["sublayers"].items(), key=lambda kv: int(kv[1]["flat_idx"]))
    mask = np.zeros((T, len(items)), dtype=bool)
    for _, entry in items:
        sidx = int(entry["flat_idx"])
        row = entry.get("expanded_mask")
        if row is None:
            row = build_expanded_mask_for_sublayer(entry["zones"], T).tolist()
        mask[:, sidx] = np.asarray(row, dtype=bool)
    return mask


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a.astype(np.float64), 0.0, 1.0)


def _stats_dict(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 sub-layer scheduler synthesis for DiT")
    parser.add_argument("--stage0-dir", default="dit_s3cache/stage0/stage0_output_sublayer")
    parser.add_argument("--output-dir", default="dit_s3cache/stage1/stage1_output_sublayer")
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--smooth-window", "--sw", dest="smooth_window", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_l1", type=float, default=1.0)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--min-zone-len", type=int, default=2)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--K-values", default="10,15,20,25")
    parser.add_argument("--sw-values", default="2,3,5")
    parser.add_argument("--lambda-values", default="0.5,1.0")
    args = parser.parse_args()

    if args.sweep:
        reports = run_sweep(
            stage0_dir=args.stage0_dir,
            output_root=args.output_dir,
            K_values=_parse_int_list(args.K_values),
            smooth_windows=_parse_int_list(args.sw_values),
            lambda_values=_parse_float_list(args.lambda_values),
            k_min=args.k_min,
            k_max=args.k_max,
            min_zone_len=args.min_zone_len,
        )
        print(f"Done sweep: {args.output_dir} ({len(reports)} runs)")
    else:
        run_stage1_sublayer(
            stage0_dir=args.stage0_dir,
            output_dir=args.output_dir,
            K=args.K,
            smooth_window=args.smooth_window,
            lambda_l1=args.lambda_l1,
            k_min=args.k_min,
            k_max=args.k_max,
            min_zone_len=args.min_zone_len,
        )
        print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
