#!/usr/bin/env python3
"""DiT sub-layer Stage2 cache schedule -> FID run.

This is the sub-layer counterpart of ``sample_stage2_cache_scheduler_dit.py``.
Jobs consume ``cache_schedule_sublayer.json`` files produced by
``stage2_to_dit_cache_sublayer.py``.
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
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dit_s3cache.fid.fid_cache_sensitivity import (
    clear_generated,
    compute_fid_adm,
    create_npz_from_sample_folder,
    load_everything,
    make_cfg_inputs,
)
from dit_s3cache.stage2.sub_layer.stage2_runtime_refine_sublayer_dit import (
    DiTStage2SubLayerContext,
)
from dit_s3cache.start_run.stage2_to_dit_cache_sublayer import load_cache_schedule


BRANCHES = ("msa", "mlp")
EXPECTED_NUM_BLOCKS = 28
EXPECTED_NUM_SUBLAYERS = EXPECTED_NUM_BLOCKS * len(BRANCHES)


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", s)


def _repo_rel_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def _get_git_commit() -> str | None:
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


def _append_runs_index(index_path: Path, entry: dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _fid_index_key(num_images: int) -> str:
    if num_images == 5000:
        return "FID@5K"
    if num_images == 50000:
        return "FID@50K"
    return f"FID@{num_images}"


def _round_fid_index(x: float | None) -> float | None:
    if x is None:
        return None
    return round(float(x), 3)


def _compact_run_index_id(start_dt: datetime, scheduler_name: str) -> str:
    return f"{start_dt.strftime('%m%d%H%M%S')}__{_sanitize_name(scheduler_name)}"


def default_results_root() -> Path:
    return _REPO_ROOT / "dit_s3cache/results/fid_dit_stage2_sublayer"


def default_jobs() -> list[tuple[str, str]]:
    rel = "dit_s3cache/stage2/stage2_output_sublayer/src_sweep_K20_sw3_lam1/cache_schedule_sublayer.json"
    return [("sweep_K20_sw3_lam1_sublayer", rel)]


def make_run_subdirectory(results_root: Path, scheduler_name: str) -> Path:
    start = datetime.now()
    date_str = start.strftime("%Y%m%d")
    time_str = start.strftime("%m%d_%H")
    safe = _sanitize_name(scheduler_name)
    run_dir = results_root / date_str / safe / f"{time_str}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_fid_args_for_run(run_dir: Path, common: argparse.Namespace) -> argparse.Namespace:
    gen_dir = run_dir / "_scratch_gen_png"
    sample_npz = run_dir / "_scratch_samples.npz"
    gen_dir.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(
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


def configure_run_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("dit_start_run_sublayer")
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


def _load_cache_payload(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "dit_s3cache_sublayer_recompute_v1":
        raise ValueError(f"Unexpected sub-layer cache schedule format: {payload.get('format')!r}")
    if int(payload.get("T", -1)) <= 0:
        raise ValueError("cache schedule payload must contain positive T")
    return payload


def _raw_schedule_to_step_schedule(
    raw_schedule: dict[tuple[int, str], set[int]],
    raw_t_to_step_idx: dict[int, int],
) -> dict[tuple[int, str], set[int]]:
    out: dict[tuple[int, str], set[int]] = {}
    for key, raw_steps in raw_schedule.items():
        missing = sorted(int(t) for t in raw_steps if int(t) not in raw_t_to_step_idx)
        if missing:
            raise ValueError(f"{key}: raw timesteps not in sampling schedule: {missing[:8]}")
        out[key] = {int(raw_t_to_step_idx[int(t)]) for t in raw_steps}
    expected = {(b, branch) for b in range(EXPECTED_NUM_BLOCKS) for branch in BRANCHES}
    if set(out) != expected:
        raise ValueError("sub-layer schedule does not cover all 28 blocks x 2 branches")
    return out


def _zero_sublayer_cache_stats() -> dict[str, Any]:
    return {
        "per_block": {
            f"block_{b:02d}": {
                "msa_cache_hits": 0,
                "msa_recompute_hits": 0,
                "mlp_cache_hits": 0,
                "mlp_recompute_hits": 0,
            }
            for b in range(EXPECTED_NUM_BLOCKS)
        }
    }


def _merge_sublayer_cache_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for block_name, stats in src.get("per_block", {}).items():
        cur = dst.setdefault("per_block", {}).setdefault(block_name, {})
        for key, val in stats.items():
            cur[key] = int(cur.get(key, 0)) + int(val)


def _baseline_full_compute_schedule_stats(T: int) -> dict[str, Any]:
    total = int(T * EXPECTED_NUM_SUBLAYERS)
    return {
        "T": int(T),
        "n_sublayers": EXPECTED_NUM_SUBLAYERS,
        "total_recompute_count": total,
        "total_cache_count": 0,
        "recompute_ratio": 1.0,
        "cache_ratio": 0.0,
        "per_sublayer": {
            f"block_{b:02d}_{branch}": {
                "recompute_count": int(T),
                "cache_count": 0,
                "recompute_ratio": 1.0,
                "cache_ratio": 0.0,
            }
            for b in range(EXPECTED_NUM_BLOCKS)
            for branch in BRANCHES
        },
    }


def _format_fr_grid_sublayer(
    raw_schedule: dict[tuple[int, str], set[int]],
    *,
    sampling_timesteps: list[int],
) -> dict[str, Any]:
    lines: list[str] = []
    by_layer: dict[str, str] = {}
    name_w = len("block_00_msa")
    for b in range(EXPECTED_NUM_BLOCKS):
        for branch in BRANCHES:
            name = f"block_{b:02d}_{branch}"
            rt_set = raw_schedule.get((b, branch), set())
            row = " ".join(f"{('F' if int(t) in rt_set else 'R'):^2}" for t in sampling_timesteps)
            by_layer[name] = row
            lines.append(f"{name.ljust(name_w)} : {row}")
    return {
        "meta": {
            "source": "dit_sublayer_cache_schedule_raw_timesteps",
            "columns_left_to_right": "denoising order: sampling_timesteps[0]=first_step",
            "sampling_timesteps_left_to_right": [int(x) for x in sampling_timesteps],
            "T": len(sampling_timesteps),
            "n_sublayers": EXPECTED_NUM_SUBLAYERS,
            "cell_token": "F=full_compute(recompute) R=cache_reuse",
        },
        "by_layer": by_layer,
        "aligned_text_block": "\n".join(lines),
    }


def generate_images_sublayer(
    *,
    model: torch.nn.Module,
    diffusion: Any,
    vae: Any,
    args: argparse.Namespace,
    device: str,
    step_schedule: dict[tuple[int, str], set[int]] | None,
    raw_t_to_step_idx: dict[int, int],
    experiment_name: str,
) -> dict[str, Any]:
    from dit_s3cache.fid.fid_cache_sensitivity import seed_all

    seed_all(args.seed)
    latent_size = args.image_size // 8
    saved = 0
    total_batches = math.ceil(args.num_fid_samples / args.per_side_batch_size)
    iterator = range(total_batches)
    if args.progress:
        iterator = tqdm(iterator, desc=f"Generating [{experiment_name}]")

    aggregate = _zero_sublayer_cache_stats()
    use_cache = step_schedule is not None

    for _ in iterator:
        current_bs = min(args.per_side_batch_size, args.num_fid_samples - saved)
        if current_bs <= 0:
            break

        z, y = make_cfg_inputs(
            batch_size=current_bs,
            latent_size=latent_size,
            num_classes=args.num_classes,
            device=device,
        )
        model_kwargs = {"y": y, "cfg_scale": args.cfg_scale}
        if use_cache:
            assert step_schedule is not None
            with DiTStage2SubLayerContext(
                model=model,
                recompute_step_indices_by_sublayer=step_schedule,
                raw_t_to_step_idx=raw_t_to_step_idx,
                callback=None,
                cache_enabled=True,
            ) as ctx:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                )
                _merge_sublayer_cache_stats(aggregate, ctx.aggregate_stats())
        else:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )

        samples, _ = samples.chunk(2, dim=0)
        decoded = vae.decode(samples / 0.18215).sample
        decoded = (
            torch.clamp(127.5 * decoded + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        for sample_idx, sample in enumerate(decoded):
            Image.fromarray(sample).save(args.gen_image_dir / f"{saved + sample_idx:06d}.png")
        saved += decoded.shape[0]

        del z, y, samples, decoded
        torch.cuda.empty_cache()

    if saved != args.num_fid_samples:
        raise RuntimeError(f"Generated {saved} samples, expected {args.num_fid_samples}")
    return aggregate if use_cache else {}


def generate_and_compute_fid_sublayer(
    *,
    model: torch.nn.Module,
    diffusion: Any,
    vae: Any,
    args: argparse.Namespace,
    device: str,
    step_schedule: dict[tuple[int, str], set[int]] | None,
    raw_t_to_step_idx: dict[int, int],
    experiment_name: str,
) -> dict[str, Any]:
    clear_generated(args.gen_image_dir)
    args.sample_npz.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_npz.exists():
        args.sample_npz.unlink()

    cache_stats = generate_images_sublayer(
        model=model,
        diffusion=diffusion,
        vae=vae,
        args=args,
        device=device,
        step_schedule=step_schedule,
        raw_t_to_step_idx=raw_t_to_step_idx,
        experiment_name=experiment_name,
    )
    create_npz_from_sample_folder(
        args.gen_image_dir,
        args.sample_npz,
        args.num_fid_samples,
        experiment_name=experiment_name,
    )
    fid_value, evaluator_output = compute_fid_adm(
        adm_python=args.adm_python,
        adm_evaluator=args.adm_evaluator,
        ref_batch=args.ref_batch,
        sample_batch=args.sample_npz,
    )
    if not args.keep_generated:
        clear_generated(args.gen_image_dir)
    return {
        "fid": fid_value,
        "num_fid_samples": int(args.num_fid_samples),
        "model": str(args.model),
        "image_size": int(args.image_size),
        "num_sampling_steps": int(args.num_sampling_steps),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "vae": str(args.vae),
        "gen_image_dir": str(args.gen_image_dir),
        "sample_npz": str(args.sample_npz),
        "ref_batch": str(args.ref_batch),
        "adm_evaluator": str(args.adm_evaluator),
        "adm_output": evaluator_output,
        "cache_stats": cache_stats,
    }


def _record_run_failed(
    *,
    manifest: dict[str, Any],
    run_dir: Path,
    start_dt: datetime,
    scheduler_name: str,
    runs_index_path: Path | None,
    common: argparse.Namespace,
) -> None:
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


def run_one_job(
    *,
    scheduler_name: str,
    cache_schedule_json: str | None,
    results_root: Path,
    common: argparse.Namespace,
    runs_index_path: Path | None,
    log: logging.Logger,
) -> None:
    start_dt = datetime.now()
    start_iso = start_dt.isoformat(timespec="seconds")
    run_id = f"{start_dt.strftime('%Y%m%dT%H%M%S')}__{_sanitize_name(scheduler_name)}"
    run_dir = make_run_subdirectory(results_root, scheduler_name)
    log_path = run_dir / "run.log"

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger = logging.getLogger("dit_start_run_sublayer")
    root_logger.addHandler(fh)

    manifest: dict[str, Any] = {}
    try:
        use_cache = cache_schedule_json is not None
        cache_path = _resolve_repo_path(cache_schedule_json).resolve() if use_cache else None
        manifest = {
            "run_id": run_id,
            "status": "running",
            "start_time": start_iso,
            "end_time": None,
            "duration_sec": None,
            "scheduler_name": scheduler_name,
            "cache_schedule_path": str(cache_path) if use_cache else None,
            "num_images": int(common.num_fid_samples),
            "seed": int(common.seed),
            "repo": "dit_s3cache/start_run/sample_stage2_cache_scheduler_sublayer_dit.py",
            "command_argv": sys.argv[:],
            "git_commit": _get_git_commit(),
            "hostname": socket.gethostname(),
            "output_dir": str(run_dir.resolve()),
        }
        _write_json(run_dir / "run_manifest.json", manifest)

        device = common.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            raise RuntimeError("DiT sub-layer FID start_run expects CUDA.")
        torch.backends.cuda.matmul.allow_tf32 = bool(common.tf32)
        torch.backends.cudnn.allow_tf32 = bool(common.tf32)
        torch.set_grad_enabled(False)

        fid_args = build_fid_args_for_run(run_dir, common)
        model, diffusion, vae = load_everything(fid_args, device)
        sampling_timesteps = list(reversed(diffusion.timestep_map))[: int(common.num_sampling_steps)]
        T = len(sampling_timesteps)
        raw_t_to_step_idx = {int(t): i for i, t in enumerate(sampling_timesteps)}
        if len(raw_t_to_step_idx) != T:
            raise ValueError("sampling_timesteps must be injective")

        step_schedule: dict[tuple[int, str], set[int]] | None = None
        raw_schedule: dict[tuple[int, str], set[int]] | None = None
        cache_payload: dict[str, Any] | None = None
        schedule_stats: dict[str, Any]

        if use_cache:
            assert cache_path is not None
            cache_payload = _load_cache_payload(cache_path)
            if int(cache_payload["T"]) != T:
                raise ValueError(f"cache schedule T={cache_payload['T']} != diffusion steps T={T}")
            payload_timesteps = [int(x) for x in cache_payload.get("sampling_timesteps", [])]
            if payload_timesteps and payload_timesteps != [int(x) for x in sampling_timesteps]:
                raise ValueError("cache schedule sampling_timesteps do not match current diffusion")
            raw_schedule = load_cache_schedule(cache_path)
            step_schedule = _raw_schedule_to_step_schedule(raw_schedule, raw_t_to_step_idx)
            schedule_stats = copy.deepcopy(cache_payload.get("summary", {}))
            log.info("Sub-layer cache enabled | recompute_ratio=%s", schedule_stats.get("recompute_ratio"))
        else:
            schedule_stats = _baseline_full_compute_schedule_stats(T)
            log.info("Baseline path: full recompute all sub-layers")

        run_result = generate_and_compute_fid_sublayer(
            model=model,
            diffusion=diffusion,
            vae=vae,
            args=fid_args,
            device=device,
            step_schedule=step_schedule,
            raw_t_to_step_idx=raw_t_to_step_idx,
            experiment_name=scheduler_name,
        )
        score = float(run_result["fid"])

        npz_path = Path(str(run_result.get("sample_npz", fid_args.sample_npz)))
        if npz_path.is_file():
            npz_path.unlink()
        scratch_gen = Path(str(run_result.get("gen_image_dir", fid_args.gen_image_dir)))
        if scratch_gen.is_dir():
            for p_item in scratch_gen.glob("*"):
                p_item.unlink(missing_ok=True)

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

        summary_obj = {
            "run_id": run_id,
            "status": "success",
            "scheduler_name": scheduler_name,
            "cache_schedule_path": str(cache_path) if use_cache else None,
            "scheduler_config_path": cache_payload.get("scheduler_json") if cache_payload else None,
            "num_images": int(common.num_fid_samples),
            "seed": int(common.seed),
            "fid_5k": score,
            "recompute_ratio": schedule_stats.get("recompute_ratio"),
            "cache_ratio": schedule_stats.get("cache_ratio"),
            "total_recompute_count": schedule_stats.get("total_recompute_count"),
            "total_cache_count": schedule_stats.get("total_cache_count"),
            "n_sublayers": schedule_stats.get("n_sublayers"),
            "start_time": start_iso,
            "end_time": end_dt.isoformat(timespec="seconds"),
            "duration_sec": duration_sec,
            "detail_stats_path": str((run_dir / "detail_stats.json").resolve()),
        }
        _write_json(run_dir / "summary.json", summary_obj)

        detail_stats = {
            "run_id": run_id,
            "scheduler_name": scheduler_name,
            "per_sublayer": schedule_stats.get("per_sublayer", {}),
            "raw_estimation_stats": schedule_stats,
            "fid_run_meta": {
                k: run_result[k]
                for k in ("cache_stats", "adm_output", "ref_batch")
                if k in run_result
            },
        }
        if raw_schedule is not None:
            detail_stats["effective_scheduler_fr_grid"] = _format_fr_grid_sublayer(
                raw_schedule, sampling_timesteps=[int(x) for x in sampling_timesteps]
            )
        _write_json(run_dir / "detail_stats.json", detail_stats)

        if cache_payload is not None:
            _write_json(run_dir / "cache_schedule_sublayer.snapshot.json", cache_payload)
            scheduler_json = cache_payload.get("scheduler_json")
            if scheduler_json and Path(str(scheduler_json)).is_file():
                with open(scheduler_json, "r", encoding="utf-8") as f:
                    _write_json(run_dir / "scheduler_config.snapshot.json", json.load(f))

        if runs_index_path is not None:
            fk = _fid_index_key(int(common.num_fid_samples))
            _append_runs_index(
                runs_index_path,
                {
                    "rid": _compact_run_index_id(start_dt, scheduler_name),
                    fk: _round_fid_index(score),
                    "d": start_dt.strftime("%Y%m%d"),
                    "sch": scheduler_name,
                    "r": schedule_stats.get("recompute_ratio"),
                    "seed": int(common.seed),
                    "out": _repo_rel_path(run_dir),
                    "sum": _repo_rel_path(run_dir / "summary.json"),
                    "st": "success",
                },
            )
        log.info("[done] %s | FID@%s=%s | out=%s", scheduler_name, common.num_fid_samples, score, run_dir)

    except BaseException:
        if run_dir.is_dir() and (run_dir / "run_manifest.json").is_file():
            _record_run_failed(
                manifest=manifest,
                run_dir=run_dir,
                start_dt=start_dt,
                scheduler_name=scheduler_name,
                runs_index_path=runs_index_path,
                common=common,
            )
        raise
    finally:
        root_logger.removeHandler(fh)
        fh.close()


def parse_jobs(cli_jobs: list[list[str]] | None, log: logging.Logger | None = None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]]
    if not cli_jobs:
        pairs = [(n, rel) for n, rel in default_jobs()]
        if log is not None:
            log.info("No --job supplied; default sub-layer schedule path: %s", pairs[0][1])
    else:
        pairs = []
        for row in cli_jobs:
            if len(row) != 2:
                raise ValueError("--job expects NAME CACHE_SCHEDULE_JSON pairs")
            pairs.append((row[0], row[1]))

    validated: list[tuple[str, str]] = []
    for name, jpath in pairs:
        jp = _resolve_repo_path(jpath)
        if not jp.is_file():
            raise FileNotFoundError(f"cache schedule JSON not found: {jp}")
        validated.append((name, str(jp)))
    return validated


def validate_cache_schedule_file(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path).resolve()
    payload = _load_cache_payload(cache_path)
    raw_schedule = load_cache_schedule(cache_path)
    sampling_timesteps = [int(x) for x in payload.get("sampling_timesteps", [])]
    if len(sampling_timesteps) != int(payload["T"]):
        raise ValueError(f"{cache_path}: sampling_timesteps length does not match T")
    raw_t_to_step_idx = {int(t): i for i, t in enumerate(sampling_timesteps)}
    if len(raw_t_to_step_idx) != len(sampling_timesteps):
        raise ValueError(f"{cache_path}: sampling_timesteps must be injective")
    step_schedule = _raw_schedule_to_step_schedule(raw_schedule, raw_t_to_step_idx)
    return {
        "path": str(cache_path),
        "T": int(payload["T"]),
        "n_sublayers": len(step_schedule),
        "recompute_ratio": payload.get("summary", {}).get("recompute_ratio"),
        "cache_ratio": payload.get("summary", {}).get("cache_ratio"),
        "scheduler_json": payload.get("scheduler_json"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="DiT sub-layer Stage2 cache schedule -> FID@5K (ADM).")
    p.add_argument("--base", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--job", action="append", nargs=2, metavar=("SCHEDULER_NAME", "CACHE_SCHEDULE_JSON"))
    p.add_argument("--results-root", type=str, default=str(default_results_root()))
    p.add_argument("--runs-index-path", type=str, default=None)
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate cache schedule jobs and exit without loading DiT or running FID.",
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
    p.add_argument("--adm-evaluator", type=str, default=str(fid_dir / "evaluator.py"))
    p.add_argument("--ref-batch", type=str, default=str(fid_dir / "VIRTUAL_imagenet256_labeled.npz"))
    p.add_argument("--adm-python", type=str, default=sys.executable)

    args = p.parse_args()
    results_root = _resolve_repo_path(args.results_root)
    runs_index = Path(args.runs_index_path).resolve() if args.runs_index_path else results_root / "runs_index.jsonl"

    common_ns = argparse.Namespace(**{k: getattr(args, k) for k in vars(args)})

    jobs_to_run: list[tuple[str, str | None]] = []
    if args.base:
        jobs_to_run.append(("dit_baseline_full_compute_sublayer", None))
    jobs_to_run.extend(parse_jobs(args.job))

    if args.validate_only:
        validations = []
        for sch_name, jpath in jobs_to_run:
            if jpath is None:
                validations.append({"scheduler_name": sch_name, "baseline": True})
            else:
                item = validate_cache_schedule_file(jpath)
                item["scheduler_name"] = sch_name
                validations.append(item)
        print(json.dumps({"status": "ok", "jobs": validations}, indent=2, ensure_ascii=False))
        return

    results_root.mkdir(parents=True, exist_ok=True)
    slog = configure_run_logging(results_root / "_driver.log")
    if not args.job:
        slog.info("No --job supplied; default sub-layer schedule path: %s", default_jobs()[0][1])
    slog.info("results_root=%s runs_index=%s jobs=%d", results_root, runs_index, len(jobs_to_run))

    for sch_name, jpath in jobs_to_run:
        run_one_job(
            scheduler_name=sch_name,
            cache_schedule_json=jpath,
            results_root=results_root,
            common=common_ns,
            runs_index_path=runs_index,
            log=slog,
        )
    slog.info("All sub-layer jobs complete. runs_index: %s", runs_index)


if __name__ == "__main__":
    main()
