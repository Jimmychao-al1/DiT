"""
DiT S3-Cache Stage 0 for sub-layer evidence.

Loads ``dit_s3cache_sublayer_v1`` evidence, normalizes the 56 MSA/MLP evidence
curves, and expands block-level FID weights so each block's MSA and MLP share
the same sensitivity weight.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from dit_s3cache.evidence.hooks_sublayer import BRANCHES, sublayer_name
from dit_s3cache.stage0.stage0_dit import (
    _DEFAULT_EPS_NOISE_DIT,
    _DEFAULT_QUANTILE_FID,
    _K_VALUES_DIT,
    compute_fid_weights,
    load_dit_delta_fid,
    normalize_minmax,
    save_object_npy_numpy1_compat,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("Stage0DiTSubLayer")

_DEFAULT_N_BLOCKS = 28
_DEFAULT_N_SUBLAYERS = 56
_DEFAULT_N_STEPS = 250


def expected_sublayer_names(n_blocks: int) -> list[str]:
    return [
        sublayer_name(block_idx, branch)
        for block_idx in range(n_blocks)
        for branch in BRANCHES
    ]


def load_dit_sublayer_evidence(
    npz_path: str,
    expected_n_sublayers: int = _DEFAULT_N_SUBLAYERS,
    expected_n_steps: int = _DEFAULT_N_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[str], int, int]:
    """Load sub-layer evidence and return interval-wise arrays."""

    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evidence NPZ 不存在: {path}")

    data = np.load(path, allow_pickle=True)
    required = (
        "l1_diff_sublayer",
        "cos_sim_sublayer",
        "grassmann_dist_sublayer",
        "pair_count_sublayer",
        "sub_layer_names",
    )
    for key in required:
        if key not in data:
            raise KeyError(f"Sub-layer evidence NPZ 缺少必要 key: {key!r}")

    l1_full = data["l1_diff_sublayer"].astype(np.float64)
    cos_full = data["cos_sim_sublayer"].astype(np.float64)
    svd_full = data["grassmann_dist_sublayer"].astype(np.float64)
    pair_count = data["pair_count_sublayer"]
    names = [str(x) for x in data["sub_layer_names"].tolist()]

    S, T = l1_full.shape
    LOGGER.info(f"Sub-layer Evidence NPZ 載入：S={S}, T={T}")
    if S != expected_n_sublayers:
        LOGGER.warning(f"S={S} 與預期 {expected_n_sublayers} 不符，繼續執行")
    if T != expected_n_steps:
        LOGGER.warning(f"T={T} 與預期 {expected_n_steps} 不符，繼續執行")
    if cos_full.shape != (S, T) or svd_full.shape != (S, T) or pair_count.shape != (S, T):
        raise ValueError(
            "Array shapes 不一致："
            f"l1={l1_full.shape}, cos={cos_full.shape}, svd={svd_full.shape}, pair={pair_count.shape}"
        )
    if len(names) != S:
        raise ValueError(f"sub_layer_names length {len(names)} != S={S}")

    pc0 = pair_count[:, 0]
    if np.any(pc0 != 0):
        LOGGER.warning(f"pair_count[:, 0] 不全為 0（{np.sum(pc0 != 0)} 個 sub-layer 例外）")
    else:
        LOGGER.info("✅ pair_count[:, 0] 全為 0（step_idx=0 確認無 pair）")

    l1_interval = l1_full[:, 1:]
    cosdist_interval = 1.0 - cos_full[:, 1:]
    svd_interval = svd_full[:, 1:]

    evidence_meta: dict[str, Any] = {}
    if "metadata_json" in data:
        try:
            raw = data["metadata_json"]
            meta_str = str(raw) if raw.ndim == 0 else str(raw.item())
            evidence_meta = json.loads(meta_str)
        except Exception as e:
            LOGGER.warning(f"metadata_json 解析失敗（繼續執行）: {e}")

    return l1_interval, cosdist_interval, svd_interval, pair_count, evidence_meta, names, S, T


def expand_block_fid_weights_to_sublayers(
    sublayer_names: list[str],
    fid_json: str,
    k_values: list[int],
    eps_noise: float,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute 28 block-level FID weights and expand them to sub-layer order."""

    fid_block_names, delta_fid, baseline_fid = load_dit_delta_fid(fid_json, k_values=k_values)
    block_names = [f"block_{i}" for i in range(_DEFAULT_N_BLOCKS)]

    missing = [b for b in block_names if b not in delta_fid]
    if missing:
        LOGGER.warning(f"以下 {len(missing)} 個 block 在 FID JSON 中找不到 (S_b=0)：{missing}")
    extra = [b for b in fid_block_names if b not in set(block_names)]
    if extra:
        LOGGER.warning(f"FID JSON 中有額外 block（不在 DiT block_0..27 內，忽略）：{extra}")

    w_block_clip, w_block_rank = compute_fid_weights(
        block_names=block_names,
        delta_fid=delta_fid,
        k_values=k_values,
        eps_noise=eps_noise,
        quantile=quantile,
    )

    w_clip = np.zeros(len(sublayer_names), dtype=np.float32)
    w_rank = np.zeros(len(sublayer_names), dtype=np.float32)
    for i, name in enumerate(sublayer_names):
        block_idx = _parse_zero_padded_sublayer_block_idx(name)
        w_clip[i] = w_block_clip[block_idx]
        w_rank[i] = w_block_rank[block_idx]
    return w_clip, w_rank, baseline_fid


def run_stage0_dit_sublayer(
    evidence_npz: str,
    fid_json: str,
    output_dir: str,
    eps_noise: float = _DEFAULT_EPS_NOISE_DIT,
    quantile: float = _DEFAULT_QUANTILE_FID,
    k_values: list[int] | None = None,
) -> None:
    if k_values is None:
        k_values = _K_VALUES_DIT

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 80)
    LOGGER.info("DiT S3-Cache Stage 0: Sub-layer Normalization + FID Weights")
    LOGGER.info("=" * 80)

    LOGGER.info("\n[Step 1] 載入 Sub-layer Evidence NPZ...")
    l1_int, cosdist_int, svd_int, pair_count, evidence_meta, sublayer_names, S, T = (
        load_dit_sublayer_evidence(evidence_npz)
    )
    interval_len = T - 1

    expected_names = expected_sublayer_names(S // len(BRANCHES))
    if sublayer_names != expected_names:
        raise ValueError(
            "sub_layer_names 順序不符合 block-major zero-padded convention: "
            f"head={sublayer_names[:4]}, expected={expected_names[:4]}"
        )

    pre_norm_stats = {
        "l1": _stats(l1_int, "l1_diff"),
        "cosdist": _stats(cosdist_int, "1-cos_sim"),
        "svd": _stats(svd_int, "grassmann_dist"),
    }
    for s in pre_norm_stats.values():
        LOGGER.info(
            f"  Pre-norm {s['name']}: min={s['min']:.6f}, max={s['max']:.6f}, "
            f"mean={s['mean']:.6f}, std={s['std']:.6f}, NaN={s['n_nan']}, Inf={s['n_inf']}"
        )

    LOGGER.info("\n[Step 2] Min-max 正規化（三種指標獨立處理）...")
    l1_norm = normalize_minmax(l1_int)
    cosdist_norm = normalize_minmax(cosdist_int)
    svd_norm = normalize_minmax(svd_int)

    LOGGER.info("\n[Step 3] 載入 block-level FID sensitivity 並展開至 sub-layer...")
    w_clip, w_rank, baseline_fid = expand_block_fid_weights_to_sublayers(
        sublayer_names=sublayer_names,
        fid_json=fid_json,
        k_values=k_values,
        eps_noise=eps_noise,
        quantile=quantile,
    )

    t_curr_interval = np.arange(T - 2, -1, -1, dtype=np.int32)
    axis_def_str = (
        "sub-layer interval-wise: analysis interval index j (0..T-2) keeps internal order; "
        "display label is t_curr=(T-2)-j"
    )

    LOGGER.info("\n[Step 4] 數值有效性驗證...")
    _check_array(l1_norm, "l1_interval_norm")
    _check_array(cosdist_norm, "cosdist_interval_norm")
    _check_array(svd_norm, "svd_interval_norm")
    _check_array(w_clip, "fid_w_clip")
    _check_array(w_rank, "fid_w_rank")

    LOGGER.info(f"\n[Step 5] 存檔至 {out_path}...")
    block_names_arr = np.array(sublayer_names, dtype=object)
    save_object_npy_numpy1_compat(out_path / "block_names.npy", block_names_arr)
    save_object_npy_numpy1_compat(out_path / "sub_layer_names.npy", block_names_arr)
    np.save(out_path / "l1_interval_norm.npy", l1_norm)
    np.save(out_path / "cosdist_interval_norm.npy", cosdist_norm)
    np.save(out_path / "svd_interval_norm.npy", svd_norm)
    np.save(out_path / "fid_w_clip.npy", w_clip)
    np.save(out_path / "fid_w_rank.npy", w_rank)
    np.save(out_path / "fid_weights.npy", w_clip)
    np.save(out_path / "fid_w_qdiffae_clip.npy", w_clip)
    np.save(out_path / "fid_w_qdiffae_rank.npy", w_rank)
    np.save(out_path / "t_curr_interval.npy", t_curr_interval)
    save_object_npy_numpy1_compat(
        out_path / "axis_interval_def.npy",
        np.array(axis_def_str, dtype=object),
    )

    metadata = {
        "stage": "DiT S3-Cache Stage 0 Sub-layer",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": evidence_meta.get("format", "dit_s3cache_sublayer_v1"),
        "model": evidence_meta.get("model", "DiT-XL/2"),
        "image_size": evidence_meta.get("image_size", 256),
        "n_blocks": S // len(BRANCHES),
        "n_sub_layers": S,
        "branches": list(BRANCHES),
        "n_steps": T,
        "interval_len": interval_len,
        "evidence_npz": str(Path(evidence_npz).resolve()),
        "fid_json": str(Path(fid_json).resolve()),
        "fid_weight_granularity": "block-level FID sensitivity expanded to msa/mlp within each block",
        "k_values_fid": k_values,
        "eps_noise": eps_noise,
        "quantile": quantile,
        "baseline_fid": baseline_fid,
        "l1_denominator_mode": evidence_meta.get("l1_denominator", "unknown"),
        "l1_source_key": "l1_diff_sublayer[:, 1:]",
        "cos_source_key": "1 - cos_sim_sublayer[:, 1:]",
        "svd_source_key": "grassmann_dist_sublayer[:, 1:]",
        "t_axis_definition": axis_def_str,
        "pair_count_step0_nonzero": int(np.sum(pair_count[:, 0] != 0)),
        "pre_norm_stats": pre_norm_stats,
        "output_files": [
            "block_names.npy",
            "sub_layer_names.npy",
            "l1_interval_norm.npy",
            "cosdist_interval_norm.npy",
            "svd_interval_norm.npy",
            "fid_w_clip.npy",
            "fid_w_rank.npy",
            "fid_weights.npy",
            "fid_w_qdiffae_clip.npy",
            "fid_w_qdiffae_rank.npy",
            "axis_interval_def.npy",
            "t_curr_interval.npy",
            "stage0_metadata.json",
        ],
    }
    with open(out_path / "stage0_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    LOGGER.info("\n✅ Sub-layer Stage 0 完成。")
    LOGGER.info(f"  Sub-layers:   {S}")
    LOGGER.info(f"  T:            {T}  (intervals={interval_len})")
    LOGGER.info(f"  Baseline FID: {baseline_fid:.6f}")
    LOGGER.info(f"  Output:       {out_path}")


def _parse_zero_padded_sublayer_block_idx(name: str) -> int:
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "block" or parts[2] not in BRANCHES:
        raise ValueError(f"Invalid sub-layer name: {name!r}")
    return int(parts[1])


def _stats(arr: np.ndarray, name: str) -> dict[str, float | int | str]:
    valid = arr[np.isfinite(arr)]
    return {
        "name": name,
        "min": float(valid.min()) if len(valid) else float("nan"),
        "max": float(valid.max()) if len(valid) else float("nan"),
        "mean": float(valid.mean()) if len(valid) else float("nan"),
        "std": float(valid.std()) if len(valid) else float("nan"),
        "n_nan": int(np.isnan(arr).sum()),
        "n_inf": int(np.isinf(arr).sum()),
    }


def _check_array(arr: np.ndarray, name: str) -> None:
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    in_range = bool((arr >= 0).all() and (arr <= 1).all())
    status = "✅" if (not has_nan and not has_inf and in_range) else "❌"
    LOGGER.info(
        f"  {status} {name}: shape={arr.shape}, NaN={has_nan}, Inf={has_inf}, "
        f"in[0,1]={in_range}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"
    )
    if not in_range or has_nan or has_inf:
        raise ValueError(f"{name} contains invalid normalized values")


if __name__ == "__main__":
    _repo = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="DiT S3-Cache Stage 0: Sub-layer evidence normalization + FID weights"
    )
    parser.add_argument(
        "--evidence-npz",
        type=str,
        default=str(_repo / "dit_s3cache/outputs/evidence_dit_xl2_256_sublayer.npz"),
    )
    parser.add_argument(
        "--fid-json",
        type=str,
        default=str(_repo / "dit_s3cache/fid/fid_sensitivity_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_repo / "dit_s3cache/stage0/stage0_output_sublayer"),
    )
    parser.add_argument("--eps-noise", type=float, default=_DEFAULT_EPS_NOISE_DIT)
    parser.add_argument("--quantile", type=float, default=_DEFAULT_QUANTILE_FID)
    parser.add_argument("--k-values", nargs="+", type=int, default=_K_VALUES_DIT)
    args = parser.parse_args()

    run_stage0_dit_sublayer(
        evidence_npz=args.evidence_npz,
        fid_json=args.fid_json,
        output_dir=args.output_dir,
        eps_noise=args.eps_noise,
        quantile=args.quantile,
        k_values=args.k_values,
    )
