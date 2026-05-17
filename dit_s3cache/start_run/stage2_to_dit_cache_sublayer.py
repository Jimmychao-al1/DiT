"""Convert sub-layer Stage2 scheduler JSON to runtime recompute schedules.

The runtime wrapper ``CachedDiTBlockSubLayer`` expects:

    Dict[Tuple[int, str], Set[int]]

where values are raw model timesteps that should be recomputed.  This adapter
stores those sets in JSON using keys such as ``"0_msa"`` and ``"0_mlp"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from diffusion import create_diffusion

from dit_s3cache.stage2.sub_layer.stage2_dit_sublayer import (
    load_sublayer_scheduler_config,
    parse_sublayer_name,
)


BRANCHES = ("msa", "mlp")


def sampling_timesteps_for_dit(num_sampling_steps: int, T: int) -> list[int]:
    diffusion = create_diffusion(str(num_sampling_steps))
    sampling_timesteps = list(reversed(diffusion.timestep_map))[:T]
    if len(sampling_timesteps) != T:
        raise ValueError(f"sampling timestep length {len(sampling_timesteps)} != T={T}")
    if len(set(sampling_timesteps)) != len(sampling_timesteps):
        raise ValueError("sampling timesteps must be injective")
    return [int(x) for x in sampling_timesteps]


def scheduler_config_to_recompute_schedule(
    cfg: dict[str, Any],
    sampling_timesteps: list[int],
) -> dict[tuple[int, str], set[int]]:
    T = int(cfg["T"])
    if len(sampling_timesteps) != T:
        raise ValueError(f"sampling_timesteps length {len(sampling_timesteps)} != T={T}")

    out: dict[tuple[int, str], set[int]] = {}
    for name, entry in sorted(cfg["sublayers"].items(), key=lambda kv: int(kv[1]["flat_idx"])):
        block_idx, branch = parse_sublayer_name(name)
        row = np.asarray(entry["expanded_mask"], dtype=bool)
        if row.shape != (T,):
            raise ValueError(f"{name}: expanded_mask shape {row.shape} != ({T},)")
        if not bool(row[0]):
            raise ValueError(f"{name}: first step must be COMPUTE")
        recompute = {
            int(sampling_timesteps[step_idx])
            for step_idx, should_compute in enumerate(row.tolist())
            if bool(should_compute)
        }
        out[(block_idx, branch)] = recompute

    expected_keys = {(b, branch) for b in range(28) for branch in BRANCHES}
    if set(out.keys()) != expected_keys:
        missing = sorted(expected_keys - set(out.keys()))
        extra = sorted(set(out.keys()) - expected_keys)
        raise ValueError(f"schedule keys mismatch: missing={missing}, extra={extra}")
    return out


def write_cache_schedule_json(
    *,
    scheduler_json: str | Path,
    output_json: str | Path,
    num_sampling_steps: int = 250,
) -> dict[str, Any]:
    cfg = load_sublayer_scheduler_config(scheduler_json)
    T = int(cfg["T"])
    sampling_timesteps = sampling_timesteps_for_dit(num_sampling_steps, T)
    schedule = scheduler_config_to_recompute_schedule(cfg, sampling_timesteps)

    schedules_json = {
        f"{block_idx}_{branch}": sorted(int(x) for x in steps)
        for (block_idx, branch), steps in sorted(schedule.items())
    }
    summary = build_summary(cfg, schedule, sampling_timesteps)
    payload = {
        "format": "dit_s3cache_sublayer_recompute_v1",
        "scheduler_json": str(Path(scheduler_json).resolve()),
        "num_sampling_steps": int(num_sampling_steps),
        "T": int(T),
        "timestep_semantics": "values are raw model timesteps from reversed diffusion.timestep_map",
        "schedule_semantics": "recompute_schedules; listed timesteps are COMPUTE, complement is CACHE",
        "sampling_timesteps": sampling_timesteps,
        "recompute_schedules": schedules_json,
        "summary": summary,
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    return payload


def load_cache_schedule(json_path: str | Path) -> dict[tuple[int, str], set[int]]:
    """Load adapter JSON into the recompute-set format expected by the runtime wrapper."""

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    schedules = payload.get("recompute_schedules")
    if not isinstance(schedules, dict):
        raise ValueError("cache schedule JSON missing recompute_schedules")

    out: dict[tuple[int, str], set[int]] = {}
    for key, values in schedules.items():
        block_str, branch = key.split("_", 1)
        block_idx = int(block_str)
        if branch not in BRANCHES:
            raise ValueError(f"invalid branch in schedule key {key!r}")
        out[(block_idx, branch)] = {int(x) for x in values}

    expected_keys = {(b, branch) for b in range(28) for branch in BRANCHES}
    if set(out.keys()) != expected_keys:
        missing = sorted(expected_keys - set(out.keys()))
        extra = sorted(set(out.keys()) - expected_keys)
        raise ValueError(f"schedule keys mismatch: missing={missing}, extra={extra}")
    return out


def build_summary(
    cfg: dict[str, Any],
    schedule: dict[tuple[int, str], set[int]],
    sampling_timesteps: list[int],
) -> dict[str, Any]:
    T = int(cfg["T"])
    first_raw_t = int(sampling_timesteps[0])
    per_sublayer: dict[str, Any] = {}
    msa_cache_ratios = []
    mlp_cache_ratios = []

    for (block_idx, branch), recompute in sorted(schedule.items()):
        name = f"block_{block_idx:02d}_{branch}"
        recompute_count = len(recompute)
        cache_count = T - recompute_count
        cache_ratio = cache_count / T
        per_sublayer[name] = {
            "recompute_count": int(recompute_count),
            "cache_count": int(cache_count),
            "recompute_ratio": float(recompute_count / T),
            "cache_ratio": float(cache_ratio),
            "first_raw_t_is_compute": bool(first_raw_t in recompute),
        }
        if branch == "msa":
            msa_cache_ratios.append(cache_ratio)
        else:
            mlp_cache_ratios.append(cache_ratio)

    total_recompute = sum(len(v) for v in schedule.values())
    total_cells = T * len(schedule)
    return {
        "n_sublayers": int(len(schedule)),
        "T": int(T),
        "total_recompute_count": int(total_recompute),
        "total_cache_count": int(total_cells - total_recompute),
        "recompute_ratio": float(total_recompute / total_cells),
        "cache_ratio": float(1.0 - total_recompute / total_cells),
        "msa_mean_cache_ratio": float(np.mean(msa_cache_ratios)),
        "mlp_mean_cache_ratio": float(np.mean(mlp_cache_ratios)),
        "first_raw_t": first_raw_t,
        "first_raw_t_cache_entries": [
            f"{b}_{branch}" for (b, branch), steps in sorted(schedule.items()) if first_raw_t not in steps
        ],
        "per_sublayer": per_sublayer,
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
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
    parser = argparse.ArgumentParser(description="Convert sub-layer Stage2 scheduler to runtime recompute JSON")
    parser.add_argument("--scheduler-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    args = parser.parse_args()

    payload = write_cache_schedule_json(
        scheduler_json=args.scheduler_json,
        output_json=args.output_json,
        num_sampling_steps=args.num_sampling_steps,
    )
    summary = payload["summary"]
    print(
        "Done: {out} | cache_ratio={cache:.4f} "
        "msa_cache={msa:.4f} mlp_cache={mlp:.4f}".format(
            out=args.output_json,
            cache=summary["cache_ratio"],
            msa=summary["msa_mean_cache_ratio"],
            mlp=summary["mlp_mean_cache_ratio"],
        )
    )


if __name__ == "__main__":
    main()
