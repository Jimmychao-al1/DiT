#!/usr/bin/env python3
"""
DiT-XL/2 Stage2 refined scheduler → FID@5K（ADM evaluator），對齊 Diff-AE ``start_run`` 產物格式。

Baseline（無 cache）：每 block 對所有 model timestep full recompute。
Cache：由 ``stage2_to_dit_cache.stage2_json_to_dit_cache_scheduler`` 自 expanded_mask 轉成 raw_t set。

結果目錄：``dit_s3cache/results/fid_dit_stage2/YYYYMMDD/sch_name/MMdd_HH_sch_name/``
產物：``run_manifest.json``, ``summary.json``, ``detail_stats.json``,
「scheduler_config.snapshot.json」（best-effort）、``run.log``。
不重保留 ``sample_npz`` / PNG scratch（跑完即刪 scratch）。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import socket
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dit_s3cache.fid.fid_cache_sensitivity import (
    generate_and_compute_fid,
    load_everything,
)
from dit_s3cache.stage2.stage2_scheduler_adapter_dit import (
    EXPECTED_NUM_BLOCKS,
    TIME_ORDER_EXPECTED,
    load_stage1_scheduler_config,
)
from dit_s3cache.start_run.stage2_to_dit_cache import (
    load_scheduler_and_build_cache_scheduler,
)


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", s)


def _compact_run_index_id(start_dt: datetime, scheduler_name: str) -> str:
    return f"{start_dt.strftime('%m%d%H%M%S')}__{_sanitize_name(scheduler_name)}"


def _fid_index_key(num_images: int) -> str:
    if num_images == 5000:
        return "FID@5K"
    if num_images == 50000:
        return "FID@50K"
    return f"FID@{num_images}"


def _round_fid_index(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), 3)


def _repo_rel_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def _get_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(_REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def _append_runs_index(index_path: Path, entry: Dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _compute_schedule_stats_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """純 expanded_mask → full/reuse counts（不依賴 raw_t）；與 Diff-AE start_run 語意對齊。"""
    T = int(cfg["T"])
    blocks: List[Dict[str, Any]] = cfg.get("blocks", [])
    shared_zones: List[Dict[str, Any]] = cfg.get("shared_zones", [])

    zone_ts: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        ts, te = int(z["t_start"]), int(z["t_end"])
        zone_ts[zid] = list(range(te, ts + 1))

    total_full = 0
    total_reuse = 0
    per_block_recompute: Dict[str, int] = {}
    per_block_reuse: Dict[str, int] = {}
    full_compute_blocks_count = 0

    per_zone: Dict[str, Dict[str, Any]] = {
        str(zid): {"full_count": 0, "reuse_count": 0, "num_timesteps_in_zone": len(ts)}
        for zid, ts in zone_ts.items()
    }

    for b in blocks:
        rt = str(b.get("runtime_name") or b.get("name") or f"block_{b.get('id', '?')}")
        mask = b.get("expanded_mask", [])
        n_full = sum(1 for v in mask if v)
        n_reuse = T - n_full
        total_full += n_full
        total_reuse += n_reuse
        per_block_recompute[rt] = n_full
        per_block_reuse[rt] = n_reuse
        if n_full == T:
            full_compute_blocks_count += 1

        for zid, ts_list in zone_ts.items():
            zkey = str(zid)
            for t_val in ts_list:
                si = (T - 1) - int(t_val)
                if 0 <= si < len(mask):
                    if mask[si]:
                        per_zone[zkey]["full_count"] += 1
                    else:
                        per_zone[zkey]["reuse_count"] += 1

    num_cells = T * len(blocks) if blocks else 1
    full_compute_ratio = round(total_full / num_cells, 6)

    return {
        "T": T,
        "num_blocks": len(blocks),
        "total_full_compute_count": total_full,
        "total_cache_reuse_count": total_reuse,
        "full_compute_ratio": full_compute_ratio,
        "full_compute_blocks_count": full_compute_blocks_count,
        "per_block_recompute_count": per_block_recompute,
        "per_block_reuse_count": per_block_reuse,
        "per_block_recompute_ratio": {
            rt: round(cnt / T, 6) for rt, cnt in per_block_recompute.items()
        },
        "per_zone_recompute_stats": per_zone,
    }


def _baseline_full_compute_schedule_stats(T: int) -> Dict[str, Any]:
    return {
        "T": int(T),
        "num_blocks": EXPECTED_NUM_BLOCKS,
        "total_full_compute_count": int(T * EXPECTED_NUM_BLOCKS),
        "total_cache_reuse_count": 0,
        "full_compute_ratio": 1.0,
        "full_compute_blocks_count": int(EXPECTED_NUM_BLOCKS),
        "per_block_recompute_count": {
            f"block_{i}": T for i in range(EXPECTED_NUM_BLOCKS)
        },
        "per_block_reuse_count": {f"block_{i}": 0 for i in range(EXPECTED_NUM_BLOCKS)},
        "per_block_recompute_ratio": {f"block_{i}": 1.0 for i in range(EXPECTED_NUM_BLOCKS)},
        "per_zone_recompute_stats": {},
    }


def _load_stage2_sidecar_stats(scheduler_json_path: Path) -> Dict[str, Any]:
    summary_p = scheduler_json_path.parent / "stage2_refinement_summary.json"
    if not summary_p.is_file():
        return {}
    try:
        data = json.loads(summary_p.read_text(encoding="utf-8"))
        zone_adj = data.get("zone_k_adjustments", [])
        peak_adj = data.get("peak_mask_adjustments", [])
        per_zone_adj: Dict[str, int] = {}
        for entry in zone_adj:
            zid = str(entry.get("zone_id", "?"))
            per_zone_adj[zid] = per_zone_adj.get(zid, 0) + 1
        return {
            "zone_adjustments_count": len(zone_adj),
            "peak_adjustments_count": len(peak_adj),
            "per_zone_adjustment_stats": per_zone_adj,
        }
    except Exception:
        return {}


def _format_fr_grid_dit(
    cache_sched: Dict[int, Set[int]],
    *,
    sampling_timesteps: List[int],
) -> Dict[str, Any]:
    """raw_t 序與採樣一致：左→右為 denoising 第一步到最後一步。"""
    lines: List[str] = []
    name_w = max(len(f"block_{i}") for i in range(EXPECTED_NUM_BLOCKS))
    by_layer: Dict[str, str] = {}
    for bi in range(EXPECTED_NUM_BLOCKS):
        row = []
        rt_set = cache_sched.get(bi, set())
        for t in sampling_timesteps:
            tok = "F" if int(t) in rt_set else "R"
            row.append(f"{tok:^2}")
        srow = " ".join(row)
        name = f"block_{bi}"
        by_layer[name] = srow
        lines.append(f"{name.ljust(name_w)} : {srow}")

    return {
        "meta": {
            "source": "dit_cache_scheduler_raw_timesteps_after_stage2_conversion",
            "columns_left_to_right": "denoising order: sampling_timesteps[0]=first_step",
            "sampling_timesteps_left_to_right": [int(x) for x in sampling_timesteps],
            "T": len(sampling_timesteps),
            "num_blocks": EXPECTED_NUM_BLOCKS,
            "cell_token": "F=full_compute(recompute) R=cache_reuse",
        },
        "by_layer": by_layer,
        "aligned_text_block": "\n".join(lines),
    }


def default_results_root() -> Path:
    return _REPO_ROOT / "dit_s3cache/results/fid_dit_stage2"


def default_jobs() -> List[tuple[str, str]]:
    rel = (
        "dit_s3cache/stage2/stage2_output/src_baseline_p2_K20_sw3_lam1.0/"
        "02_refined_blockwise/stage2_refined_scheduler_config.json"
    )
    return [("baseline_p2_K20_sw3_lam1.0_blockwise", rel)]


def make_run_subdirectory(results_root: Path, scheduler_name: str) -> Path:
    start = datetime.now()
    date_str = start.strftime("%Y%m%d")
    time_str = start.strftime("%m%d_%H")
    safe = _sanitize_name(scheduler_name)
    run_dir = results_root / date_str / safe / f"{time_str}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_fid_args_for_run(run_dir: Path, common: argparse.Namespace) -> argparse.Namespace:
    """fid_cache_sensitivity.generate_and_compute_fid 所需 Namespace。"""
    gen_dir = run_dir / "_scratch_gen_png"
    sample_npz = run_dir / "_scratch_samples.npz"
    gen_dir.mkdir(parents=True, exist_ok=True)

    ns = argparse.Namespace(
        adm_evaluator=Path(common.adm_evaluator).resolve(),
        ref_batch=Path(common.ref_batch).resolve(),
        adm_python=str(common.adm_python),
        gen_image_dir=gen_dir,
        sample_npz=sample_npz,
        keep_generated=False,
        num_fid_samples=int(common.num_fid_samples),
        model=str(common.model),
        image_size=int(common.image_size),
        num_classes=int(common.num_classes),
        num_sampling_steps=int(common.num_sampling_steps),
        per_side_batch_size=int(common.per_side_batch_size),
        cfg_scale=float(common.cfg_scale),
        seed=int(common.seed),
        vae=str(common.vae),
        ckpt=common.ckpt,
        tf32=bool(common.tf32),
        progress=bool(common.progress),
    )
    return ns


def configure_run_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("dit_start_run")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.propagate = False
    return log


def run_one_job(
    *,
    scheduler_name: str,
    scheduler_json: Optional[str],
    results_root: Path,
    common: argparse.Namespace,
    runs_index_path: Optional[Path],
    log: logging.Logger,
) -> None:
    """執行單一 FID run（baseline 時 scheduler_json 為 None）。"""
    start_dt = datetime.now()
    start_iso = start_dt.isoformat(timespec="seconds")
    run_id = f"{start_dt.strftime('%Y%m%dT%H%M%S')}__{_sanitize_name(scheduler_name)}"
    run_dir = make_run_subdirectory(results_root, scheduler_name)
    log_path = run_dir / "run.log"

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger = logging.getLogger("dit_start_run")
    root_logger.addHandler(fh)

    try:
        use_cache = scheduler_json is not None
        sched_path_resolved = (
            _resolve_repo_path(scheduler_json).resolve() if use_cache else None
        )

        manifest: Dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "start_time": start_iso,
            "end_time": None,
            "duration_sec": None,
            "scheduler_name": scheduler_name,
            "scheduler_config_path": str(sched_path_resolved) if use_cache else None,
            "num_images": int(common.num_fid_samples),
            "seed": int(common.seed),
            "repo": "dit_s3cache/start_run/sample_stage2_cache_scheduler_dit.py",
            "command_argv": sys.argv[:],
            "git_commit": _get_git_commit(),
            "hostname": socket.gethostname(),
            "output_dir": str(run_dir.resolve()),
        }
        _write_json(run_dir / "run_manifest.json", manifest)

        device = common.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        if device == "cpu":
            raise RuntimeError("DiT FID start_run expects CUDA.")

        torch_mod = __import__("torch")
        torch_mod.backends.cuda.matmul.allow_tf32 = bool(common.tf32)
        torch_mod.backends.cudnn.allow_tf32 = bool(common.tf32)
        torch_mod.set_grad_enabled(False)

        fid_args = build_fid_args_for_run(run_dir, common)
        model, diffusion, vae = load_everything(fid_args, device)

        sampling_timesteps = list(reversed(diffusion.timestep_map))[
            : int(common.num_sampling_steps)
        ]
        T = len(sampling_timesteps)
        log.info("sampling_timesteps len=%d first=%s last=%s", T, sampling_timesteps[0], sampling_timesteps[-1])

        cache_sched: Optional[Dict[int, Set[int]]] = None
        cfg_snapshot: Optional[Dict[str, Any]] = None
        sidecar: Dict[str, Any] = {}

        if use_cache:
            assert sched_path_resolved is not None
            cfg = load_stage1_scheduler_config(sched_path_resolved)
            if int(cfg["T"]) != T:
                raise ValueError(
                    f"Scheduler T={cfg['T']} != diffusion steps T={T} (check --num-sampling-steps)"
                )
            if cfg.get("time_order") != TIME_ORDER_EXPECTED:
                raise ValueError(f"Expected time_order {TIME_ORDER_EXPECTED!r}, got {cfg.get('time_order')!r}")
            cache_sched = load_scheduler_and_build_cache_scheduler(
                sched_path_resolved,
                sampling_timesteps,
                validate_cfg=True,
            )
            cfg_snapshot = copy.deepcopy(cfg)
            sidecar = _load_stage2_sidecar_stats(sched_path_resolved)
            log.info("Stage2 cache enabled | full_compute_ratio(disk)=%s", _compute_schedule_stats_from_cfg(cfg).get("full_compute_ratio"))
        else:
            cache_sched = None
            cfg_snapshot = {
                "note": "dit_baseline_full_compute: no Stage2 JSON; every block recomputes every timestep",
                "T": T,
                "time_order": TIME_ORDER_EXPECTED,
                "num_blocks": EXPECTED_NUM_BLOCKS,
            }
            log.info("Baseline path: full recompute all blocks (no cache reuse)")

        score: Optional[float] = None
        run_result: Dict[str, Any] = {}
        try:
            run_result = generate_and_compute_fid(
                model=model,
                diffusion=diffusion,
                vae=vae,
                args=fid_args,
                device=device,
                cache_scheduler=cache_sched,
                experiment_name=scheduler_name,
            )
            score = float(run_result["fid"])

            npz_path = Path(str(run_result.get("sample_npz", fid_args.sample_npz)))
            if npz_path.is_file():
                npz_path.unlink()
                log.info("Removed scratch NPZ: %s", npz_path)

            scratch_gen = Path(str(run_result.get("gen_image_dir", fid_args.gen_image_dir)))
            if scratch_gen.is_dir():
                for p_item in scratch_gen.glob("*"):
                    p_item.unlink(missing_ok=True)

        except BaseException:
            end_dt_failed = datetime.now()
            manifest.update(
                {
                    "status": "failed",
                    "end_time": end_dt_failed.isoformat(timespec="seconds"),
                    "duration_sec": round((end_dt_failed - start_dt).total_seconds(), 2),
                    "error_message": str(sys.exc_info()[1]),
                    "traceback": traceback.format_exc(),
                }
            )
            _write_json(run_dir / "run_manifest.json", manifest)
            if runs_index_path is not None:
                fk = _fid_index_key(int(common.num_fid_samples))
                _append_runs_index(
                    runs_index_path,
                    {
                        "rid": _compact_run_index_id(start_dt, scheduler_name),
                        fk: _round_fid_index(None),
                        "d": start_dt.strftime("%Y%m%d"),
                        "sch": scheduler_name,
                        "r": None,
                        "seed": int(common.seed),
                        "out": _repo_rel_path(run_dir),
                        "sum": None,
                        "st": "failed",
                    },
                )
            raise

        end_dt = datetime.now()
        duration_sec = round((end_dt - start_dt).total_seconds(), 2)
        manifest.update(
            {
                "status": "success",
                "end_time": end_dt.isoformat(timespec="seconds"),
                "duration_sec": duration_sec,
                "fid_score": score,
            }
        )
        _write_json(run_dir / "run_manifest.json", manifest)

        if use_cache and cfg_snapshot is not None:
            sched_stats = _compute_schedule_stats_from_cfg(cfg_snapshot)
        else:
            sched_stats = _baseline_full_compute_schedule_stats(T)

        summary_obj: Dict[str, Any] = {
            "run_id": run_id,
            "status": "success",
            "scheduler_name": scheduler_name,
            "scheduler_config_path": str(sched_path_resolved) if use_cache else None,
            "threshold_config_path": None,
            "force-prefix": "F",
            "force-full-prefix-steps": 0,
            "num_images": int(common.num_fid_samples),
            "seed": int(common.seed),
            "fid_5k": float(score) if score is not None else None,
            "full_compute_ratio": sched_stats.get("full_compute_ratio"),
            "total_full_compute_count": sched_stats.get("total_full_compute_count"),
            "total_cache_reuse_count": sched_stats.get("total_cache_reuse_count"),
            "full_compute_blocks_count": sched_stats.get("full_compute_blocks_count"),
            "zone_adjustments_count": sidecar.get("zone_adjustments_count"),
            "peak_adjustments_count": sidecar.get("peak_adjustments_count"),
            "start_time": start_iso,
            "end_time": end_dt.isoformat(timespec="seconds"),
            "duration_sec": duration_sec,
            "detail_stats_path": str((run_dir / "detail_stats.json").resolve()),
        }
        _write_json(run_dir / "summary.json", summary_obj)

        detail_stats: Dict[str, Any] = {
            "run_id": run_id,
            "scheduler_name": scheduler_name,
            "per_block_recompute_count": sched_stats.get("per_block_recompute_count", {}),
            "per_block_reuse_count": sched_stats.get("per_block_reuse_count", {}),
            "per_zone_recompute_stats": sched_stats.get("per_zone_recompute_stats", {}),
            "per_zone_adjustment_stats": sidecar.get("per_zone_adjustment_stats", {}),
            "raw_estimation_stats": {
                "full_compute_ratio": sched_stats.get("full_compute_ratio"),
                "total_full_compute_count": sched_stats.get("total_full_compute_count"),
                "total_cache_reuse_count": sched_stats.get("total_cache_reuse_count"),
                "full_compute_blocks_count": sched_stats.get("full_compute_blocks_count"),
                "T": sched_stats.get("T"),
                "num_blocks": sched_stats.get("num_blocks"),
                "per_block_recompute_ratio": sched_stats.get("per_block_recompute_ratio", {}),
            },
            "fid_run_meta": {
                k: run_result[k]
                for k in ("cache_stats", "adm_output", "ref_batch")
                if k in run_result
            },
        }
        if cache_sched is not None:
            try:
                detail_stats["effective_scheduler_fr_grid"] = _format_fr_grid_dit(
                    cache_sched, sampling_timesteps=sampling_timesteps
                )
            except Exception as e:
                detail_stats["effective_scheduler_fr_grid"] = {"error": str(e)}
        _write_json(run_dir / "detail_stats.json", detail_stats)

        if cfg_snapshot is not None:
            _write_json(run_dir / "scheduler_config.snapshot.json", cfg_snapshot)

        if runs_index_path is not None:
            fk = _fid_index_key(int(common.num_fid_samples))
            _append_runs_index(
                runs_index_path,
                {
                    "rid": _compact_run_index_id(start_dt, scheduler_name),
                    fk: _round_fid_index(float(score) if score is not None else None),
                    "d": start_dt.strftime("%Y%m%d"),
                    "sch": scheduler_name,
                    "r": sched_stats.get("full_compute_ratio"),
                    "seed": int(common.seed),
                    "out": _repo_rel_path(run_dir),
                    "sum": _repo_rel_path(run_dir / "summary.json"),
                    "st": "success",
                },
            )

        log.info(
            "[done] %s | FID@%s=%s | out=%s",
            scheduler_name,
            common.num_fid_samples,
            score,
            run_dir,
        )

    finally:
        root_logger.removeHandler(fh)
        fh.close()


def parse_jobs(cli_jobs: Optional[List[List[str]]], log: logging.Logger) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    if not cli_jobs:
        pairs = [(n, rel) for n, rel in default_jobs()]
        log.info("No --job supplied; default P2 blockwise path: %s", pairs[0][1])
    else:
        for row in cli_jobs:
            if len(row) != 2:
                raise ValueError("--job expects NAME JSON_PATH pairs")
            pairs.append((row[0], row[1]))

    validated: List[tuple[str, str]] = []
    for name, jpath in pairs:
        jp = _resolve_repo_path(jpath)
        if not jp.is_file():
            raise FileNotFoundError(f"scheduler JSON not found: {jp}")
        validated.append((name, str(jp)))
    return validated


def main() -> None:
    p = argparse.ArgumentParser(
        description="DiT Stage2 → FID@5K (ADM), aligned with Diff-AE start_run artifacts.",
    )
    p.add_argument(
        "--base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, run dit_baseline_full_compute once before cache jobs (default: false).",
    )
    p.add_argument(
        "--job",
        action="append",
        nargs=2,
        metavar=("SCHEDULER_NAME", "JSON_PATH"),
        help="Repeatable. Default: P2 blockwise only.",
    )
    p.add_argument(
        "--results-root",
        type=str,
        default=str(default_results_root()),
        help="e.g. dit_s3cache/results/fid_dit_stage2/",
    )
    p.add_argument(
        "--runs-index-path",
        type=str,
        default=None,
        help="JSONL aggregate (default: <results-root>/runs_index.jsonl)",
    )

    p.add_argument("--model", type=str, default="DiT-XL/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--num-sampling-steps", type=int, default=250)
    p.add_argument("--num-fid-samples", type=int, default=5000)
    p.add_argument("--per-side-batch-size", type=int, default=32)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    fid_dir = _REPO_ROOT / "dit_s3cache/fid"
    p.add_argument(
        "--adm-evaluator",
        type=str,
        default=str(fid_dir / "evaluator.py"),
    )
    p.add_argument(
        "--ref-batch",
        type=str,
        default=str(fid_dir / "VIRTUAL_imagenet256_labeled.npz"),
    )
    p.add_argument("--adm-python", type=str, default=sys.executable)

    args = p.parse_args()
    results_root = _resolve_repo_path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    runs_index = (
        Path(args.runs_index_path).resolve()
        if args.runs_index_path
        else results_root / "runs_index.jsonl"
    )

    slog = configure_run_logging(results_root / "_driver.log")
    common_ns = argparse.Namespace(**{k: getattr(args, k) for k in vars(args)})
    setattr(common_ns, "num_fid_samples", args.num_fid_samples)

    jobs_to_run: List[tuple[Optional[str], Optional[str]]] = []
    if args.base:
        jobs_to_run.append(("dit_baseline_full_compute", None))
    jobs_to_run.extend(parse_jobs(args.job, slog))

    slog.info("results_root=%s runs_index=%s jobs=%d", results_root, runs_index, len(jobs_to_run))

    for sch_name, jpath in jobs_to_run:
        run_one_job(
            scheduler_name=sch_name or "unknown",
            scheduler_json=jpath,
            results_root=results_root,
            common=common_ns,
            runs_index_path=runs_index,
            log=slog,
        )

    slog.info("All jobs complete. runs_index: %s", runs_index)


if __name__ == "__main__":
    main()
