#!/usr/bin/env python3
"""
將 Stage 1 sweep 目錄下的 scheduler JSON 彙整為 CSV（與 LDM 實驗
stage1_output_ldm/csv_exports 相同欄位與邏輯）。

每個 sweep 子目錄需含：
  scheduler_config.json
  scheduler_diagnostics.json
  verification_summary.json

輸出（預設 dit_s3cache/stage1/stage1_output/csv_exports/）：
  all/                     — 全部 sweep 子目錄
  K_sw/                    — lambda == 1.0 的子集（若存在）
  lambda/                  — 固定 K、sw 的 lambda 掃描子集（若存在）
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SWEEP_DIR_RE = re.compile(
    r"^sweep_K(?P<K>\d+)_sw(?P<sw>\d+)_lam(?P<lam>[\d.]+)(?:_kmax(?P<kmax>\d+))?$"
)

SUMMARY_COLUMNS = [
    "run",
    "K",
    "sw",
    "lambda",
    "T",
    "B",
    "Z",
    "zone_len_min",
    "zone_len_max",
    "zone_len_mean",
    "change_points_count",
    "merged_zones_count",
    "total_F",
    "total_R",
    "F_ratio",
    "F_mean_per_block",
    "F_std_per_block",
    "R_mean_per_block",
    "cost_mean_per_block",
    "cost_std_per_block",
    "cost_sum_all_blocks",
    "k_mean",
    "k1_count",
    "k2_count",
    "k3_count",
    "k4_count",
    "I_cut_mean",
    "I_cut_std",
    "I_l1cos_mean",
    "I_l1cos_std",
    "Delta_abs_mean",
    "Delta_abs_max",
]

PER_BLOCK_COLUMNS = [
    "run",
    "K",
    "sw",
    "lambda",
    "block_id",
    "block_name",
    "num_F",
    "num_R",
    "F_ratio_block",
    "total_cost_J_sum_zones",
    "k_per_zone",
]

PER_ZONE_COLUMNS = [
    "run",
    "K",
    "sw",
    "lambda",
    "zone_id",
    "t_start",
    "t_end",
    "length",
    "candidate_k",
    "k1_count",
    "k2_count",
    "k3_count",
    "k4_count",
    "selected_k_mean",
    "selected_J_mean",
    "selected_J_std",
    "selected_k_per_block",
]


def parse_sweep_dir_name(name: str) -> Optional[Dict[str, Any]]:
    m = SWEEP_DIR_RE.match(name)
    if not m:
        return None
    return {
        "run": name,
        "K": int(m.group("K")),
        "sw": int(m.group("sw")),
        "lambda": float(m.group("lam")),
        "kmax": int(m.group("kmax")) if m.group("kmax") else None,
    }


def load_run_triplet(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    with open(run_dir / "scheduler_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(run_dir / "scheduler_diagnostics.json", encoding="utf-8") as f:
        diag = json.load(f)
    with open(run_dir / "verification_summary.json", encoding="utf-8") as f:
        ver = json.load(f)
    return cfg, diag, ver


def _semicolon_ints(values: Sequence[int]) -> str:
    return ";".join(str(int(v)) for v in values)


def _count_k(selected: Sequence[int]) -> Tuple[int, int, int, int]:
    arr = np.asarray(selected, dtype=int)
    return (
        int(np.sum(arr == 1)),
        int(np.sum(arr == 2)),
        int(np.sum(arr == 3)),
        int(np.sum(arr == 4)),
    )


def build_summary_row(
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
    diag: Dict[str, Any],
    ver: Dict[str, Any],
) -> Dict[str, Any]:
    T = int(cfg["T"])
    blocks = cfg["blocks"]
    B = len(blocks)
    zones = cfg["shared_zones"]
    Z = len(zones)
    lengths = [int(z["length"]) for z in zones]

    per_block = ver["per_block"]
    num_F = [int(pb["num_F"]) for pb in per_block]
    costs = [float(pb["total_cost_J_sum_zones"]) for pb in per_block]

    total_F = int(sum(num_F))
    total_R = int(B * T - total_F)
    F_ratio = total_F / (B * T) if B * T else 0.0

    all_ks: List[int] = []
    for pb in per_block:
        all_ks.extend(int(k) for k in pb["k_per_zone"])

    k1, k2, k3, k4 = _count_k(all_ks)

    delta = np.asarray(diag["Delta_processing_order"], dtype=np.float64)
    icut = diag.get("I_cut_stats", {})
    il1 = diag.get("I_l1cos_stats", {})
    cps = diag.get("change_points_step_index", [])

    return {
        "run": meta["run"],
        "K": meta["K"],
        "sw": meta["sw"],
        "lambda": meta["lambda"],
        "T": T,
        "B": B,
        "Z": Z,
        "zone_len_min": min(lengths) if lengths else 0,
        "zone_len_max": max(lengths) if lengths else 0,
        "zone_len_mean": float(statistics.mean(lengths)) if lengths else 0.0,
        "change_points_count": len(cps),
        "merged_zones_count": Z,
        "total_F": total_F,
        "total_R": total_R,
        "F_ratio": round(F_ratio, 4),
        "F_mean_per_block": round(float(statistics.mean(num_F)), 2) if num_F else 0.0,
        "F_std_per_block": round(float(statistics.pstdev(num_F)), 6) if len(num_F) > 1 else 0.0,
        "R_mean_per_block": round(float(statistics.mean([T - f for f in num_F])), 2) if num_F else 0.0,
        "cost_mean_per_block": round(float(statistics.mean(costs)), 6) if costs else 0.0,
        "cost_std_per_block": round(float(statistics.pstdev(costs)), 6) if len(costs) > 1 else 0.0,
        "cost_sum_all_blocks": round(float(sum(costs)), 6),
        "k_mean": round(float(np.mean(all_ks)), 6) if all_ks else 0.0,
        "k1_count": k1,
        "k2_count": k2,
        "k3_count": k3,
        "k4_count": k4,
        "I_cut_mean": float(icut.get("mean", 0.0)),
        "I_cut_std": float(icut.get("std", 0.0)),
        "I_l1cos_mean": float(il1.get("mean", 0.0)),
        "I_l1cos_std": float(il1.get("std", 0.0)),
        "Delta_abs_mean": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "Delta_abs_max": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def build_per_block_rows(
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
    ver: Dict[str, Any],
) -> List[Dict[str, Any]]:
    T = int(cfg["T"])
    name_by_id = {int(b["id"]): str(b["name"]) for b in cfg["blocks"]}
    rows: List[Dict[str, Any]] = []
    for pb in ver["per_block"]:
        bid = int(pb["block_id"])
        num_F = int(pb["num_F"])
        num_R = int(pb["num_R"])
        rows.append(
            {
                "run": meta["run"],
                "K": meta["K"],
                "sw": meta["sw"],
                "lambda": meta["lambda"],
                "block_id": bid,
                "block_name": name_by_id.get(bid, f"block_{bid}"),
                "num_F": num_F,
                "num_R": num_R,
                "F_ratio_block": round(num_F / T, 3) if T else 0.0,
                "total_cost_J_sum_zones": round(float(pb["total_cost_J_sum_zones"]), 8),
                "k_per_zone": _semicolon_ints(pb["k_per_zone"]),
            }
        )
    return rows


def build_per_zone_rows(
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
    ver: Dict[str, Any],
) -> List[Dict[str, Any]]:
    zone_meta = {int(z["id"]): z for z in cfg["shared_zones"]}
    rows: List[Dict[str, Any]] = []
    for pz in ver["per_zone"]:
        zid = int(pz["zone_id"])
        z = zone_meta[zid]
        selected_k = [int(k) for k in pz["selected_k_per_block"]]
        selected_J = [float(j) for j in pz["selected_J_per_block"]]
        k1, k2, k3, k4 = _count_k(selected_k)
        rows.append(
            {
                "run": meta["run"],
                "K": meta["K"],
                "sw": meta["sw"],
                "lambda": meta["lambda"],
                "zone_id": zid,
                "t_start": int(z["t_start"]),
                "t_end": int(z["t_end"]),
                "length": int(z["length"]),
                "candidate_k": _semicolon_ints(pz["candidate_k"]),
                "k1_count": k1,
                "k2_count": k2,
                "k3_count": k3,
                "k4_count": k4,
                "selected_k_mean": round(float(np.mean(selected_k)), 1) if selected_k else 0.0,
                "selected_J_mean": round(float(np.mean(selected_J)), 8) if selected_J else 0.0,
                "selected_J_std": round(float(statistics.pstdev(selected_J)), 8)
                if len(selected_J) > 1
                else 0.0,
                "selected_k_per_block": _semicolon_ints(selected_k),
            }
        )
    return rows


def discover_runs(base_out: Path) -> List[Tuple[Dict[str, Any], Path]]:
    runs: List[Tuple[Dict[str, Any], Path]] = []
    for d in sorted(base_out.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_sweep_dir_name(d.name)
        if meta is None:
            continue
        needed = (
            "scheduler_config.json",
            "scheduler_diagnostics.json",
            "verification_summary.json",
        )
        if not all((d / f).exists() for f in needed):
            print(f"⚠️  跳過（缺少 JSON）: {d.name}")
            continue
        runs.append((meta, d))
    return runs


def _sort_key(meta: Dict[str, Any]) -> Tuple[Any, ...]:
    return (meta["K"], meta["sw"], meta["lambda"])


def write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"✅ {path}  ({len(rows)} rows)")


def export_subset(
    runs: List[Tuple[Dict[str, Any], Path]],
    out_dir: Path,
    tag: str,
) -> None:
    if not runs:
        return
    summary_rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []
    zone_rows: List[Dict[str, Any]] = []

    for meta, run_dir in sorted(runs, key=lambda x: _sort_key(x[0])):
        cfg, diag, ver = load_run_triplet(run_dir)
        summary_rows.append(build_summary_row(meta, cfg, diag, ver))
        block_rows.extend(build_per_block_rows(meta, cfg, ver))
        zone_rows.extend(build_per_zone_rows(meta, cfg, ver))

    n = len(summary_rows)
    write_csv(out_dir / f"stage1_sweep_summary_{tag}.csv", SUMMARY_COLUMNS, summary_rows)
    write_csv(out_dir / f"stage1_sweep_per_block_{tag}.csv", PER_BLOCK_COLUMNS, block_rows)
    write_csv(out_dir / f"stage1_sweep_per_zone_{tag}.csv", PER_ZONE_COLUMNS, zone_rows)
    print(f"   → {n} runs → {out_dir}/")


def filter_lambda_runs(
    runs: List[Tuple[Dict[str, Any], Path]],
    fixed_K: Optional[int],
    fixed_sw: Optional[int],
) -> List[Tuple[Dict[str, Any], Path]]:
    """
    Lambda 掃描子集（對齊 LDM csv_exports/lambda/）：
    1. 若未指定，優先 K∈{15,20} 且 sw=3 的全部 runs
    2. 否則單一 (K, sw) 且含多個 lambda
    """
    if fixed_K is not None and fixed_sw is not None:
        return [(m, d) for m, d in runs if m["K"] == fixed_K and m["sw"] == fixed_sw]

    ldm_style = [
        (m, d) for m, d in runs if m["K"] in (15, 20) and m["sw"] == 3
    ]
    if len(ldm_style) >= 2:
        return ldm_style

    from collections import defaultdict

    groups: Dict[Tuple[int, int], set] = defaultdict(set)
    for meta, _ in runs:
        groups[(meta["K"], meta["sw"])].add(meta["lambda"])

    candidates = [(k, sw) for (k, sw), lams in groups.items() if len(lams) >= 2]
    if not candidates:
        return []
    for prefer in ((15, 3), (20, 3)):
        if prefer in candidates:
            k, sw = prefer
            return [(m, d) for m, d in runs if m["K"] == k and m["sw"] == sw]
    k, sw = max(candidates, key=lambda ks: len(groups[ks]))
    return [(m, d) for m, d in runs if m["K"] == k and m["sw"] == sw]


def main() -> None:
    p = argparse.ArgumentParser(description="Export Stage-1 sweep JSON to CSV (DiT)")
    p.add_argument(
        "--base_out",
        type=str,
        default="dit_s3cache/stage1/stage1_output",
        help="含 sweep_* 子目錄的 Stage 1 輸出根目錄",
    )
    p.add_argument(
        "--csv_root",
        type=str,
        default="",
        help="CSV 輸出根目錄（預設 <base_out>/csv_exports）",
    )
    p.add_argument(
        "--lambda_fixed_K",
        type=int,
        default=None,
        help="lambda 子集固定 K（預設自動推斷）",
    )
    p.add_argument(
        "--lambda_fixed_sw",
        type=int,
        default=None,
        help="lambda 子集固定 smooth_window（預設自動推斷）",
    )
    args = p.parse_args()

    base_out = Path(args.base_out)
    csv_root = Path(args.csv_root) if args.csv_root else base_out / "csv_exports"

    runs = discover_runs(base_out)
    if not runs:
        raise SystemExit(f"找不到 sweep 目錄於 {base_out}")

    print(f"發現 {len(runs)} 個 sweep runs @ {base_out}")
    export_subset(runs, csv_root / "all", f"{len(runs)}runs")

    ksw = [(m, d) for m, d in runs if abs(m["lambda"] - 1.0) < 1e-9]
    if ksw:
        export_subset(ksw, csv_root / "K_sw", f"{len(ksw)}runs")

    lam_runs = filter_lambda_runs(
        runs,
        fixed_K=args.lambda_fixed_K,
        fixed_sw=args.lambda_fixed_sw,
    )
    if lam_runs:
        export_subset(
            lam_runs,
            csv_root / "lambda",
            f"lambda_{len(lam_runs)}runs",
        )

    print(f"\n完成。CSV 根目錄: {csv_root.resolve()}")


if __name__ == "__main__":
    main()
