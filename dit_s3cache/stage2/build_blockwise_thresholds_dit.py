"""
由 stage2_runtime_diagnostics.json 的 per-block 分布，建立 DiT Stage2 blockwise thresholds。

預設採 baseline_908030 規格：
- q_zone = 0.90
- q_peak = 0.80
- peak_over_zone_ratio_min = 1.3
- source = "ported_from_baseline_908030"
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

from dit_s3cache.stage2.stage2_scheduler_adapter_dit import (
    EXPECTED_NUM_BLOCKS,
    RUNTIME_LAYER_NAMES,
    runtime_block_to_stage1_name,
)
from dit_s3cache.stage2.verify_stage2_dit import verify_blockwise_threshold_config_dict

METHOD_NAME = "blockwise_quantile_v1_dit"
SOURCE_TAG_DEFAULT = "ported_from_baseline_908030"


def _finite_values_zone(per_block_zone: Dict[str, Any], runtime_name: str) -> List[float]:
    out: List[float] = []
    zone_map = per_block_zone.get(runtime_name)
    if not isinstance(zone_map, dict):
        raise KeyError(f"per_block_zone_error missing/invalid entry for {runtime_name!r}")
    for _zid, st in zone_map.items():
        if not isinstance(st, dict):
            continue
        v = float(st.get("mean_l1", float("nan")))
        if math.isfinite(v):
            out.append(v)
    return out


def _finite_values_step(per_block_step: Dict[str, Any], runtime_name: str) -> List[float]:
    out: List[float] = []
    step_map = per_block_step.get(runtime_name)
    if not isinstance(step_map, dict):
        raise KeyError(f"per_block_step_error missing/invalid entry for {runtime_name!r}")
    for _t, st in step_map.items():
        if not isinstance(st, dict):
            continue
        v = float(st.get("l1", float("nan")))
        if math.isfinite(v):
            out.append(v)
    return out


def _quantile_or_raise(vals: List[float], q: float, *, label: str) -> float:
    if not vals:
        raise ValueError(f"{label}: no finite samples to compute quantile")
    if not (0.0 <= q <= 1.0) or math.isnan(q) or math.isinf(q):
        raise ValueError(f"{label}: invalid quantile q={q!r}")
    x = float(np.quantile(np.asarray(vals, dtype=np.float64), q))
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"{label}: quantile({q}) produced non-finite value {x!r}")
    if x <= 0.0:
        raise ValueError(f"{label}: quantile({q}) must be > 0, got {x!r}")
    return x


def build_blockwise_thresholds_dit(
    *,
    diagnostics_path: Path,
    output_path: Path,
    q_zone: float,
    q_peak: float,
    peak_over_zone_ratio_min: float,
    source: str = SOURCE_TAG_DEFAULT,
) -> Dict[str, Any]:
    """
    由 Pass 1 的 stage2_runtime_diagnostics.json 計算 per-block thresholds。

    threshold 計算規則（ported from baseline_908030）：
    - zone_l1_threshold = quantile(q_zone) 對各 zone 的 mean_l1
    - peak_l1_threshold = max(quantile(q_peak) 對各 step 的 l1, peak_over_zone_ratio_min * zone_l1)
    """
    if not (peak_over_zone_ratio_min > 0.0) or math.isnan(peak_over_zone_ratio_min):
        raise ValueError(
            f"peak_over_zone_ratio_min must be finite and > 0, got {peak_over_zone_ratio_min!r}"
        )
    if not source or not str(source).strip():
        raise ValueError("source must be a non-empty string")

    with open(diagnostics_path, "r", encoding="utf-8") as f:
        diag = json.load(f)

    per_block_step = diag.get("per_block_step_error")
    per_block_zone = diag.get("per_block_zone_error")
    if not isinstance(per_block_step, dict) or not isinstance(per_block_zone, dict):
        raise ValueError(
            "diagnostics must contain per_block_step_error and per_block_zone_error objects"
        )

    per_block: List[Dict[str, Any]] = []
    for block_id, runtime_name in enumerate(RUNTIME_LAYER_NAMES):
        zvals = _finite_values_zone(per_block_zone, runtime_name)
        pvals = _finite_values_step(per_block_step, runtime_name)

        zone_thr = _quantile_or_raise(
            zvals, q_zone, label=f"block {block_id} ({runtime_name}) zone"
        )
        peak_thr = _quantile_or_raise(
            pvals, q_peak, label=f"block {block_id} ({runtime_name}) peak"
        )
        peak_thr = max(peak_thr, peak_over_zone_ratio_min * zone_thr)
        if math.isnan(peak_thr) or math.isinf(peak_thr) or peak_thr <= 0.0:
            raise ValueError(
                f"block {block_id} ({runtime_name}): invalid peak_l1_threshold {peak_thr!r}"
            )

        # DiT flat 架構：canonical_name == runtime_name == "block_X"
        canonical = runtime_block_to_stage1_name(runtime_name)
        per_block.append(
            {
                "block_id": int(block_id),
                "canonical_runtime_block_id": int(block_id),
                "canonical_name": canonical,
                "runtime_name": runtime_name,
                "num_zone_samples": int(len(zvals)),
                "num_step_samples": int(len(pvals)),
                "zone_l1_threshold": float(zone_thr),
                "peak_l1_threshold": float(peak_thr),
            }
        )

    if len(per_block) != EXPECTED_NUM_BLOCKS:
        raise RuntimeError("internal error: per_block length mismatch")

    out: Dict[str, Any] = {
        "method": METHOD_NAME,
        "source": str(source),
        "block_identity_semantics": {
            "block_id": "canonical runtime block index (0~27)",
            "canonical_runtime_block_id": "same as block_id",
        },
        "source_diagnostics_path": str(diagnostics_path.resolve()),
        "q_zone": float(q_zone),
        "q_peak": float(q_peak),
        "peak_over_zone_ratio_min": float(peak_over_zone_ratio_min),
        "per_block": per_block,
    }

    verify_blockwise_threshold_config_dict(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Stage2-DiT per-block thresholds from diagnostics JSON"
    )
    ap.add_argument(
        "--diagnostics",
        type=str,
        required=True,
        help="Path to stage2_runtime_diagnostics.json (from Pass 1 global refine)",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path, e.g. stage2_thresholds_blockwise.json",
    )
    ap.add_argument(
        "--q-zone",
        type=float,
        default=0.90,
        help="Quantile over zone mean_l1 values per block (default 0.90)",
    )
    ap.add_argument(
        "--q-peak",
        type=float,
        default=0.80,
        help="Quantile over per-step l1 values per block (default 0.80)",
    )
    ap.add_argument(
        "--peak-over-zone-ratio-min",
        type=float,
        default=1.3,
        help="Enforce peak_l1 >= this * zone_l1 (default 1.3)",
    )
    ap.add_argument(
        "--source",
        type=str,
        default=SOURCE_TAG_DEFAULT,
        help=f'Metadata source tag (default "{SOURCE_TAG_DEFAULT}")',
    )
    args = ap.parse_args()

    diagnostics_path = Path(args.diagnostics)
    output_path = Path(args.output)
    result = build_blockwise_thresholds_dit(
        diagnostics_path=diagnostics_path,
        output_path=output_path,
        q_zone=float(args.q_zone),
        q_peak=float(args.q_peak),
        peak_over_zone_ratio_min=float(args.peak_over_zone_ratio_min),
        source=str(args.source),
    )

    # 印出 threshold spread 觀察（DiT flat 架構的特有觀察點）
    zone_vals = [e["zone_l1_threshold"] for e in result["per_block"]]
    peak_vals = [e["peak_l1_threshold"] for e in result["per_block"]]
    z_min, z_max = min(zone_vals), max(zone_vals)
    p_min, p_max = min(peak_vals), max(peak_vals)
    spread = z_max / z_min if z_min > 0 else float("inf")
    print(f"Wrote {output_path.resolve()}")
    print(
        f"Zone threshold spread: min={z_min:.6f} max={z_max:.6f} ratio={spread:.2f}x "
        f"(LDM reference: ~34x)"
    )
    print(f"Peak threshold range: min={p_min:.6f} max={p_max:.6f}")


if __name__ == "__main__":
    main()
