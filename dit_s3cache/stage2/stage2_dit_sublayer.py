"""Evidence-based Stage 2 refinement for DiT sub-layer schedulers.

This sub-layer path is CPU/data driven.  It does not run DiT sampling; it uses
``l1_interval_norm`` from Stage 0 as the approximation-error proxy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from dit_s3cache.stage1.stage1_dit_sublayer import (
    build_expanded_mask_for_sublayer,
    flat_to_block_branch,
    load_stage0_sublayer,
    parse_sublayer_name,
)
from dit_s3cache.stage1.stage1_scheduler_dit import step_index_to_ddim_t


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Stage2Sub] %(message)s")
LOGGER = logging.getLogger("Stage2DiTSubLayer")

VERSION = "stage2_sublayer_refined_v1"
DEFAULT_ZONE_L1_THRESHOLD = 0.02
DEFAULT_PEAK_L1_THRESHOLD = 0.08
DEFAULT_Q_ZONE = 0.90
DEFAULT_Q_PEAK = 0.80
DEFAULT_PEAK_OVER_ZONE_RATIO_MIN = 1.3


def load_sublayer_scheduler_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"scheduler config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_sublayer_scheduler_config(cfg)
    return cfg


def validate_sublayer_scheduler_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise TypeError("scheduler config must be a dict")
    if cfg.get("granularity") != "sublayer":
        raise ValueError(f"granularity must be 'sublayer', got {cfg.get('granularity')!r}")
    if cfg.get("time_order") != "ddpm_249_to_0":
        raise ValueError(f"time_order must be 'ddpm_249_to_0', got {cfg.get('time_order')!r}")
    T = int(cfg["T"])
    sublayers = cfg.get("sublayers")
    if not isinstance(sublayers, dict) or len(sublayers) != 56:
        raise ValueError(f"sublayers must contain 56 entries, got {len(sublayers) if isinstance(sublayers, dict) else type(sublayers)}")

    seen = set()
    for name, entry in sublayers.items():
        flat_idx = int(entry["flat_idx"])
        if flat_idx in seen:
            raise ValueError(f"duplicate flat_idx: {flat_idx}")
        seen.add(flat_idx)
        block_idx, branch = parse_sublayer_name(name)
        expected_block, expected_branch = flat_to_block_branch(flat_idx)
        if (block_idx, branch) != (expected_block, expected_branch):
            raise ValueError(
                f"{name}: flat_idx={flat_idx} maps to {(expected_block, expected_branch)}, "
                f"but name maps to {(block_idx, branch)}"
            )
        row = np.asarray(entry.get("expanded_mask"), dtype=bool)
        if row.shape != (T,):
            raise ValueError(f"{name}: expanded_mask shape {row.shape} != ({T},)")
        if not bool(row[0]):
            raise ValueError(f"{name}: first step must be COMPUTE")
        _validate_zones(name, entry.get("zones"), T)
    if seen != set(range(56)):
        raise ValueError(f"flat_idx set must be 0..55, got missing={sorted(set(range(56)) - seen)}")


def error_by_step_from_stage0(stage0_dir: str | Path) -> tuple[list[str], np.ndarray]:
    names, l1, _, _, _, _, t_curr = load_stage0_sublayer(stage0_dir)
    S, Tm1 = l1.shape
    T = Tm1 + 1
    expected_t_curr = np.arange(T - 2, -1, -1, dtype=np.int32)
    if not np.array_equal(t_curr, expected_t_curr):
        raise ValueError("Stage 0 sub-layer t_curr_interval must be descending T-2..0")
    err = np.zeros((S, T), dtype=np.float64)
    # Stage 0 stores interval j for reused DDPM t=(T-2)-j.  Since step_idx=(T-1)-t,
    # interval j maps to step_idx=j+1; step_idx=0 has no prior interval.
    err[:, 1:] = l1
    err[:, 0] = 0.0
    return names, err


def build_diagnostics(cfg: dict[str, Any], error_step: np.ndarray) -> dict[str, Any]:
    T = int(cfg["T"])
    items = _sorted_sublayer_items(cfg)
    per_step: dict[str, dict[str, dict[str, float]]] = {}
    per_zone: dict[str, dict[str, dict[str, Any]]] = {}
    all_l1: list[float] = []

    for name, entry in items:
        sidx = int(entry["flat_idx"])
        per_step[name] = {}
        per_zone[name] = {}
        for step_idx in range(T):
            l1 = float(error_step[sidx, step_idx])
            per_step[name][str(step_idx)] = {
                "l1": l1,
                "ddpm_t": int(step_index_to_ddim_t(step_idx, T)),
            }
            all_l1.append(l1)
        for z in entry["zones"]:
            zid = int(z["zone_id"])
            steps = list(range(int(z["start_step"]), int(z["end_step"]) + 1))
            vals = [float(error_step[sidx, si]) for si in steps]
            per_zone[name][str(zid)] = {
                "mean_l1": float(np.mean(vals)) if vals else float("nan"),
                "max_l1": float(np.max(vals)) if vals else float("nan"),
                "num_steps": int(len(steps)),
                "start_step": int(z["start_step"]),
                "end_step": int(z["end_step"]),
                "start_t": int(z["start_t"]),
                "end_t": int(z["end_t"]),
            }

    return {
        "T": int(T),
        "granularity": "sublayer",
        "error_source": "stage0 l1_interval_norm mapped to step space; step_idx=0 has no prior interval and is 0",
        "per_sublayer_step_error": per_step,
        "per_sublayer_zone_error": per_zone,
        "global_summary": {
            "mean_l1": float(np.mean(all_l1)),
            "max_l1": float(np.max(all_l1)),
            "num_entries": int(len(all_l1)),
        },
    }


def build_sublayerwise_thresholds(
    *,
    diagnostics_path: str | Path,
    output_path: str | Path,
    q_zone: float = DEFAULT_Q_ZONE,
    q_peak: float = DEFAULT_Q_PEAK,
    peak_over_zone_ratio_min: float = DEFAULT_PEAK_OVER_ZONE_RATIO_MIN,
) -> dict[str, Any]:
    with open(diagnostics_path, "r", encoding="utf-8") as f:
        diag = json.load(f)
    per_step = diag.get("per_sublayer_step_error")
    per_zone = diag.get("per_sublayer_zone_error")
    if not isinstance(per_step, dict) or not isinstance(per_zone, dict):
        raise ValueError("diagnostics must contain per_sublayer_step_error and per_sublayer_zone_error")

    thresholds: dict[str, Any] = {}
    for name in sorted(per_step.keys(), key=lambda n: int(parse_sublayer_name(n)[0]) * 2 + (0 if parse_sublayer_name(n)[1] == "msa" else 1)):
        block_idx, branch = parse_sublayer_name(name)
        flat_idx = block_idx * 2 + (0 if branch == "msa" else 1)
        zvals = _finite_values([v.get("mean_l1") for v in per_zone.get(name, {}).values()])
        pvals = _finite_values([v.get("l1") for v in per_step.get(name, {}).values()])
        zone_thr = _quantile_or_raise(zvals, q_zone, label=f"{name} zone")
        peak_thr = _quantile_or_raise(pvals, q_peak, label=f"{name} peak")
        peak_thr = max(peak_thr, peak_over_zone_ratio_min * zone_thr)
        thresholds[name] = {
            "flat_idx": int(flat_idx),
            "block_idx": int(block_idx),
            "branch": branch,
            "num_zone_samples": int(len(zvals)),
            "num_step_samples": int(len(pvals)),
            "zone_l1_threshold": float(zone_thr),
            "peak_l1_threshold": float(peak_thr),
        }

    out = {
        "method": "sublayerwise_quantile_l1_thresholds",
        "source": "stage2_dit_sublayer",
        "source_diagnostics_path": str(Path(diagnostics_path).resolve()),
        "q_zone": float(q_zone),
        "q_peak": float(q_peak),
        "peak_over_zone_ratio_min": float(peak_over_zone_ratio_min),
        "thresholds": thresholds,
        "metadata": {
            "granularity": "sublayer",
            "n_sublayers": len(thresholds),
        },
    }
    validate_sublayer_threshold_config(out)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(_json_safe(out), f, indent=2, ensure_ascii=False)
    return out


def validate_sublayer_threshold_config(cfg: dict[str, Any]) -> None:
    thresholds = cfg.get("thresholds")
    if not isinstance(thresholds, dict) or len(thresholds) != 56:
        raise ValueError(f"thresholds must contain 56 entries, got {len(thresholds) if isinstance(thresholds, dict) else type(thresholds)}")
    for name, entry in thresholds.items():
        block_idx, branch = parse_sublayer_name(name)
        expected_flat = block_idx * 2 + (0 if branch == "msa" else 1)
        if int(entry["flat_idx"]) != expected_flat:
            raise ValueError(f"{name}: flat_idx mismatch")
        zt = float(entry["zone_l1_threshold"])
        pt = float(entry["peak_l1_threshold"])
        if not math.isfinite(zt) or zt <= 0:
            raise ValueError(f"{name}: invalid zone_l1_threshold {zt}")
        if not math.isfinite(pt) or pt <= 0:
            raise ValueError(f"{name}: invalid peak_l1_threshold {pt}")


def run_stage2_refine_sublayer(
    *,
    scheduler_config_path: str,
    stage0_dir: str,
    output_dir: str,
    pass_mode: str = "global",
    zone_l1_threshold: float = DEFAULT_ZONE_L1_THRESHOLD,
    peak_l1_threshold: float = DEFAULT_PEAK_L1_THRESHOLD,
    threshold_config_path: str | None = None,
) -> dict[str, Any]:
    if pass_mode not in ("global", "sublayerwise"):
        raise ValueError("pass_mode must be 'global' or 'sublayerwise'")

    cfg = load_sublayer_scheduler_config(scheduler_config_path)
    names, error_step = error_by_step_from_stage0(stage0_dir)
    config_names = [name for name, _ in _sorted_sublayer_items(cfg)]
    if names != config_names:
        raise ValueError(f"Stage0 names and scheduler names differ: {names[:4]} vs {config_names[:4]}")

    threshold_doc = None
    thresholds = None
    if threshold_config_path:
        with open(threshold_config_path, "r", encoding="utf-8") as f:
            threshold_doc = json.load(f)
        validate_sublayer_threshold_config(threshold_doc)
        thresholds = threshold_doc["thresholds"]

    diagnostics = build_diagnostics(cfg, error_step)
    diagnostics["scheduler_config_path"] = str(Path(scheduler_config_path).resolve())
    diagnostics["stage0_dir"] = str(Path(stage0_dir).resolve())
    diagnostics["stage2_threshold_meta"] = {
        "threshold_mode": pass_mode,
        "global_zone_l1": float(zone_l1_threshold),
        "global_peak_l1": float(peak_l1_threshold),
        "threshold_config_path": str(Path(threshold_config_path).resolve()) if threshold_config_path else None,
        "q_zone": threshold_doc.get("q_zone") if threshold_doc else None,
        "q_peak": threshold_doc.get("q_peak") if threshold_doc else None,
        "peak_over_zone_ratio_min": threshold_doc.get("peak_over_zone_ratio_min") if threshold_doc else None,
    }

    refined = copy.deepcopy(cfg)
    refined["version"] = VERSION
    refined["stage2_meta"] = diagnostics["stage2_threshold_meta"]
    original_mask = _mask_from_config(cfg)
    original_ratio = float(original_mask.mean())

    zone_adjustments: list[dict[str, Any]] = []
    peak_repairs: list[dict[str, Any]] = []
    per_sublayer_summary: dict[str, Any] = {}

    for name, entry in _sorted_sublayer_items(refined):
        sidx = int(entry["flat_idx"])
        zone_thr = (
            float(thresholds[name]["zone_l1_threshold"])
            if thresholds is not None
            else float(zone_l1_threshold)
        )
        peak_thr = (
            float(thresholds[name]["peak_l1_threshold"])
            if thresholds is not None
            else float(peak_l1_threshold)
        )

        zones = entry["zones"]
        zone_errors = diagnostics["per_sublayer_zone_error"][name]
        local_zone_adjustments = 0
        for z in zones:
            zid = int(z["zone_id"])
            mean_l1 = float(zone_errors[str(zid)]["mean_l1"])
            if math.isfinite(mean_l1) and mean_l1 > zone_thr:
                old_k = int(z["k"])
                new_k = max(1, old_k - 1)
                if new_k != old_k:
                    z["k"] = int(new_k)
                    local_zone_adjustments += 1
                    zone_adjustments.append(
                        {
                            "sublayer": name,
                            "flat_idx": int(sidx),
                            "zone_id": int(zid),
                            "k_before": int(old_k),
                            "k_after": int(new_k),
                            "mean_l1": float(mean_l1),
                            "zone_l1_threshold_used": float(zone_thr),
                            "threshold_mode": pass_mode,
                        }
                    )

        row = build_expanded_mask_for_sublayer(zones, int(refined["T"]))
        local_peak_repairs = 0
        for step_idx, val in enumerate(error_step[sidx]):
            if step_idx == 0:
                row[step_idx] = True
                continue
            if float(val) > peak_thr and not bool(row[step_idx]):
                row[step_idx] = True
                local_peak_repairs += 1
                peak_repairs.append(
                    {
                        "sublayer": name,
                        "flat_idx": int(sidx),
                        "step_index": int(step_idx),
                        "ddpm_t": int(step_index_to_ddim_t(step_idx, int(refined["T"]))),
                        "l1_error": float(val),
                        "peak_l1_threshold_used": float(peak_thr),
                        "threshold_mode": pass_mode,
                        "was_reuse_before_peak_repair": True,
                    }
                )

        entry["expanded_mask"] = row.tolist()
        per_sublayer_summary[name] = {
            "peak_repairs": int(local_peak_repairs),
            "zone_adjustments": int(local_zone_adjustments),
            "sublayer_recompute_ratio": float(row.mean()),
            "zone_l1_threshold_used": float(zone_thr),
            "peak_l1_threshold_used": float(peak_thr),
        }

    refined_mask = _mask_from_config(refined)
    refined_ratio = float(refined_mask.mean())
    refined["full_compute_ratio"] = refined_ratio

    diagnostics["refined_recompute_summary"] = {
        "original_full_compute_ratio": original_ratio,
        "refined_full_compute_ratio": refined_ratio,
        "total_peak_repairs": int(len(peak_repairs)),
        "total_zone_k_adjustments": int(len(zone_adjustments)),
    }

    summary = {
        "source_scheduler": Path(scheduler_config_path).parent.name,
        "pass": pass_mode,
        "original_full_compute_ratio": original_ratio,
        "refined_full_compute_ratio": refined_ratio,
        "total_peak_repairs": int(len(peak_repairs)),
        "total_zone_k_adjustments": int(len(zone_adjustments)),
        "zone_k_adjustments": zone_adjustments,
        "peak_mask_adjustments": peak_repairs,
        "per_sublayer_summary": per_sublayer_summary,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "stage2_runtime_diagnostics.json", diagnostics)
    _write_json(out / "stage2_refined_scheduler_config.json", refined)
    _write_json(out / "stage2_refinement_summary.json", summary)
    _write_json(
        out / "cache_runtime_overrides_run.json",
        {
            "stage": "stage2_dit_sublayer",
            "scheduler_config_path": str(Path(scheduler_config_path).resolve()),
            "stage0_dir": str(Path(stage0_dir).resolve()),
            "pass_mode": pass_mode,
            "T": int(refined["T"]),
        },
    )
    _append_scheduler_summary_csv(out.parent / "stage2_scheduler_summary.csv", summary, out)
    LOGGER.info("wrote Stage 2 sub-layer refine: %s", out)
    return {"diagnostics": diagnostics, "summary": summary, "refined_config": refined}


def _mask_from_config(cfg: dict[str, Any]) -> np.ndarray:
    T = int(cfg["T"])
    items = _sorted_sublayer_items(cfg)
    mask = np.zeros((T, len(items)), dtype=bool)
    for name, entry in items:
        mask[:, int(entry["flat_idx"])] = np.asarray(entry["expanded_mask"], dtype=bool)
    return mask


def _sorted_sublayer_items(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(cfg["sublayers"].items(), key=lambda kv: int(kv[1]["flat_idx"]))


def _validate_zones(name: str, zones: Any, T: int) -> None:
    if not isinstance(zones, list) or not zones:
        raise ValueError(f"{name}: zones must be a non-empty list")
    covered = np.zeros(T, dtype=bool)
    for z in zones:
        s0 = int(z["start_step"])
        s1 = int(z["end_step"])
        if not (0 <= s0 <= s1 < T):
            raise ValueError(f"{name}: invalid step zone [{s0},{s1}]")
        if np.any(covered[s0 : s1 + 1]):
            raise ValueError(f"{name}: overlapping zones at [{s0},{s1}]")
        covered[s0 : s1 + 1] = True
        if int(z["start_t"]) != step_index_to_ddim_t(s0, T):
            raise ValueError(f"{name}: start_t/start_step mismatch")
        if int(z["end_t"]) != step_index_to_ddim_t(s1, T):
            raise ValueError(f"{name}: end_t/end_step mismatch")
        if int(z["k"]) < 1:
            raise ValueError(f"{name}: k must be >= 1")
    if not covered.all():
        missing = np.where(~covered)[0].tolist()
        raise ValueError(f"{name}: zones do not cover all steps; missing={missing[:16]}")


def _finite_values(values: list[Any]) -> list[float]:
    out = []
    for v in values:
        x = float(v)
        if math.isfinite(x):
            out.append(x)
    return out


def _quantile_or_raise(values: list[float], q: float, *, label: str) -> float:
    if not values:
        raise ValueError(f"{label}: no finite values")
    x = float(np.quantile(np.asarray(values, dtype=np.float64), q))
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError(f"{label}: quantile({q}) must be finite and > 0, got {x}")
    return x


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, ensure_ascii=False)


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


def _append_scheduler_summary_csv(path: Path, summary: dict[str, Any], out_dir: Path) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "output_dir",
                "pass",
                "original_full_compute_ratio",
                "refined_full_compute_ratio",
                "total_peak_repairs",
                "total_zone_k_adjustments",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "output_dir": str(out_dir),
                "pass": summary["pass"],
                "original_full_compute_ratio": summary["original_full_compute_ratio"],
                "refined_full_compute_ratio": summary["refined_full_compute_ratio"],
                "total_peak_repairs": summary["total_peak_repairs"],
                "total_zone_k_adjustments": summary["total_zone_k_adjustments"],
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 sub-layer evidence-based refinement for DiT")
    parser.add_argument("--scheduler-config", default=None)
    parser.add_argument("--stage0-dir", default="dit_s3cache/stage0/stage0_output_sublayer")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pass-mode", choices=["global", "sublayerwise"], default="global")
    parser.add_argument("--zone-l1-threshold", type=float, default=DEFAULT_ZONE_L1_THRESHOLD)
    parser.add_argument("--peak-l1-threshold", type=float, default=DEFAULT_PEAK_L1_THRESHOLD)
    parser.add_argument("--threshold-config", default=None)
    parser.add_argument("--build-thresholds", action="store_true")
    parser.add_argument("--diagnostics", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--q-zone", type=float, default=DEFAULT_Q_ZONE)
    parser.add_argument("--q-peak", type=float, default=DEFAULT_Q_PEAK)
    parser.add_argument("--peak-over-zone-ratio-min", type=float, default=DEFAULT_PEAK_OVER_ZONE_RATIO_MIN)
    args = parser.parse_args()

    if args.build_thresholds:
        if not args.diagnostics or not args.output:
            parser.error("--build-thresholds requires --diagnostics and --output")
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
        parser.error("refine mode requires --scheduler-config and --output-dir")
    if args.pass_mode == "sublayerwise" and not args.threshold_config:
        parser.error("--threshold-config is required for --pass-mode sublayerwise")

    run_stage2_refine_sublayer(
        scheduler_config_path=args.scheduler_config,
        stage0_dir=args.stage0_dir,
        output_dir=args.output_dir,
        pass_mode=args.pass_mode,
        zone_l1_threshold=args.zone_l1_threshold,
        peak_l1_threshold=args.peak_l1_threshold,
        threshold_config_path=args.threshold_config,
    )
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
