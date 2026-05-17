"""Smoke tests for the sub-layer Stage1/Stage2/adapter chain."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dit_s3cache.evidence.hooks_sublayer import BRANCHES, sublayer_name  # noqa: E402
from dit_s3cache.stage0.stage0_dit import save_object_npy_numpy1_compat  # noqa: E402
from dit_s3cache.stage1.stage1_dit_sublayer import run_stage1_sublayer  # noqa: E402
from dit_s3cache.stage2.stage2_dit_sublayer import (  # noqa: E402
    build_sublayerwise_thresholds,
    run_stage2_refine_sublayer,
)
from dit_s3cache.start_run.stage2_to_dit_cache_sublayer import (  # noqa: E402
    load_cache_schedule,
    write_cache_schedule_json,
)


def _make_fake_stage0(root: pathlib.Path, *, T: int = 8) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    names = [sublayer_name(b, branch) for b in range(28) for branch in BRANCHES]
    S = len(names)
    Tm1 = T - 1
    rng = np.random.default_rng(0)
    base = rng.random((S, Tm1)).astype(np.float64) * 0.2
    # Make MSA/MLP visibly different for threshold spread.
    base[0::2] += 0.25
    base[1::2] += 0.05

    save_object_npy_numpy1_compat(root / "sub_layer_names.npy", np.asarray(names, dtype=object))
    np.save(root / "l1_interval_norm.npy", np.clip(base, 0, 1))
    np.save(root / "cosdist_interval_norm.npy", rng.random((S, Tm1)).astype(np.float64))
    np.save(root / "svd_interval_norm.npy", rng.random((S, Tm1)).astype(np.float64))
    fid = np.repeat(np.linspace(0.1, 1.0, 28), 2).astype(np.float64)
    np.save(root / "fid_w_clip.npy", fid)
    np.save(root / "t_curr_interval.npy", np.arange(T - 2, -1, -1, dtype=np.int32))
    save_object_npy_numpy1_compat(root / "axis_interval_def.npy", np.array("test", dtype=object))
    return root


def test_sublayer_stage1_stage2_adapter_chain(tmp_path: pathlib.Path) -> None:
    stage0 = _make_fake_stage0(tmp_path / "stage0", T=8)
    stage1 = tmp_path / "stage1"
    run_stage1_sublayer(
        stage0_dir=str(stage0),
        output_dir=str(stage1),
        K=3,
        smooth_window=2,
        lambda_l1=1.0,
        k_max=4,
    )

    expanded = np.load(stage1 / "expanded_mask.npy")
    assert expanded.shape == (8, 56)
    assert expanded[0].all()

    global_dir = tmp_path / "stage2" / "00_global_refine"
    run_stage2_refine_sublayer(
        scheduler_config_path=str(stage1 / "scheduler_config.json"),
        stage0_dir=str(stage0),
        output_dir=str(global_dir),
        pass_mode="global",
        zone_l1_threshold=0.20,
        peak_l1_threshold=0.30,
    )
    assert (global_dir / "stage2_runtime_diagnostics.json").exists()

    threshold_path = tmp_path / "stage2" / "01_sublayerwise_threshold" / "stage2_thresholds_sublayerwise.json"
    thresholds = build_sublayerwise_thresholds(
        diagnostics_path=global_dir / "stage2_runtime_diagnostics.json",
        output_path=threshold_path,
        q_zone=0.5,
        q_peak=0.5,
        peak_over_zone_ratio_min=1.1,
    )
    assert len(thresholds["thresholds"]) == 56
    assert thresholds["thresholds"]["block_00_msa"]["zone_l1_threshold"] != thresholds["thresholds"]["block_00_mlp"]["zone_l1_threshold"]

    refined_dir = tmp_path / "stage2" / "02_refined_sublayerwise"
    run_stage2_refine_sublayer(
        scheduler_config_path=str(global_dir / "stage2_refined_scheduler_config.json"),
        stage0_dir=str(stage0),
        output_dir=str(refined_dir),
        pass_mode="sublayerwise",
        threshold_config_path=str(threshold_path),
    )

    adapter_json = tmp_path / "cache_schedule_sublayer.json"
    payload = write_cache_schedule_json(
        scheduler_json=refined_dir / "stage2_refined_scheduler_config.json",
        output_json=adapter_json,
        num_sampling_steps=8,
    )
    schedule = load_cache_schedule(adapter_json)
    assert len(schedule) == 56
    assert (0, "msa") in schedule
    assert payload["summary"]["first_raw_t_cache_entries"] == []

    with open(adapter_json, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["format"] == "dit_s3cache_sublayer_recompute_v1"
