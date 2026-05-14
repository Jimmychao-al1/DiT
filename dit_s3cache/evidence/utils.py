"""Evidence metrics and accumulators for DiT S3-Cache Stage 0."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_similarity(
    residual_current: torch.Tensor,
    residual_prev: torch.Tensor,
    denominator: str = "current",
) -> dict[str, float]:
    """Compute cross-timestep residual similarity.

    Args:
        residual_current: Current residual, shape ``(N, tokens, hidden)``.
        residual_prev: Previous timestep residual with the same shape.
        denominator: Relative L1 denominator. ``"current"`` matches the DiT
            evidence spec; ``"symmetric"`` mirrors the Diff-AE/LDM L1 metric.
    """

    r_curr = residual_current.detach().float().flatten(1)
    r_prev = residual_prev.detach().float().flatten(1).to(r_curr.device)
    diff_l1 = (r_curr - r_prev).abs().sum(dim=1)

    if denominator == "current":
        denom = r_curr.abs().sum(dim=1)
    elif denominator == "previous":
        denom = r_prev.abs().sum(dim=1)
    elif denominator == "max":
        denom = torch.maximum(r_curr.abs().sum(dim=1), r_prev.abs().sum(dim=1))
    elif denominator == "symmetric":
        denom = 0.5 * (r_curr.abs().sum(dim=1) + r_prev.abs().sum(dim=1))
    else:
        raise ValueError(f"Unknown L1 denominator: {denominator}")

    l1_diff = diff_l1 / denom.clamp(min=1e-8)
    cos_sim = F.cosine_similarity(r_curr, r_prev, dim=1)

    return {
        "l1_diff": float(l1_diff.mean().item()),
        "cos_sim": float(cos_sim.mean().item()),
    }


def compute_svd_drift(
    residual_current: torch.Tensor,
    v_prev: torch.Tensor | None,
    k: int = 16,
    token_subsample: int | None = None,
) -> dict[str, Any]:
    """Compute top-k residual subspace drift via randomized low-rank SVD."""

    if k <= 0:
        raise ValueError("k must be positive")

    hidden_size = residual_current.shape[-1]
    matrix = residual_current.detach().reshape(-1, hidden_size).float()
    if token_subsample is not None and token_subsample > 0 and matrix.shape[0] > token_subsample:
        indices = torch.randperm(matrix.shape[0], device=matrix.device)[:token_subsample]
        matrix = matrix.index_select(0, indices)

    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    q = min(k, matrix.shape[0], matrix.shape[1])
    if q <= 0:
        raise ValueError(f"Cannot compute SVD for residual matrix shape {tuple(matrix.shape)}")

    _, singular_values, v = torch.svd_lowrank(matrix, q=q)
    v_current = v[:, :q].T.contiguous()  # torch.svd_lowrank returns V as (hidden, q).

    spectrum = singular_values[:q].detach().cpu().numpy()
    if q < k:
        spectrum = np.pad(spectrum, (0, k - q), constant_values=np.nan)

    if v_prev is None:
        return {
            "grassmann_dist": np.nan,
            "subspace_overlap": np.nan,
            "sv_spectrum": spectrum,
            "v_current": v_current.detach(),
        }

    v_prev = v_prev.detach().to(device=v_current.device, dtype=v_current.dtype)
    common_k = min(v_current.shape[0], v_prev.shape[0])
    cross = v_current[:common_k] @ v_prev[:common_k].T
    sigma = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
    angles = torch.acos(sigma.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

    return {
        "grassmann_dist": float(angles.norm().item()),
        "subspace_overlap": float((sigma.sum() / common_k).item()),
        "sv_spectrum": spectrum,
        "v_current": v_current.detach(),
    }


@dataclass
class EvidenceAccumulator:
    """NaN-safe accumulator for per-block, per-step evidence."""

    n_blocks: int
    n_steps: int
    k_svd: int
    l1_sum: np.ndarray = field(init=False)
    cos_sum: np.ndarray = field(init=False)
    grassmann_sum: np.ndarray = field(init=False)
    overlap_sum: np.ndarray = field(init=False)
    sv_sum: np.ndarray = field(init=False)
    pair_count: np.ndarray = field(init=False)
    sv_count: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.n_blocks, self.n_steps)
        self.l1_sum = np.zeros(shape, dtype=np.float64)
        self.cos_sum = np.zeros(shape, dtype=np.float64)
        self.grassmann_sum = np.zeros(shape, dtype=np.float64)
        self.overlap_sum = np.zeros(shape, dtype=np.float64)
        self.sv_sum = np.zeros((self.n_blocks, self.n_steps, self.k_svd), dtype=np.float64)
        self.pair_count = np.zeros(shape, dtype=np.int64)
        self.sv_count = np.zeros(shape, dtype=np.int64)

    def add_similarity(self, block_idx: int, step_idx: int, metrics: dict[str, float]) -> None:
        if np.isnan(metrics["l1_diff"]) or np.isnan(metrics["cos_sim"]):
            return
        self.l1_sum[block_idx, step_idx] += metrics["l1_diff"]
        self.cos_sum[block_idx, step_idx] += metrics["cos_sim"]
        self.pair_count[block_idx, step_idx] += 1

    def add_svd(self, block_idx: int, step_idx: int, metrics: dict[str, Any]) -> None:
        spectrum = np.asarray(metrics["sv_spectrum"], dtype=np.float64)
        valid_spectrum = np.nan_to_num(spectrum, nan=0.0)
        self.sv_sum[block_idx, step_idx] += valid_spectrum
        self.sv_count[block_idx, step_idx] += 1

        if np.isnan(metrics["grassmann_dist"]) or np.isnan(metrics["subspace_overlap"]):
            return
        self.grassmann_sum[block_idx, step_idx] += metrics["grassmann_dist"]
        self.overlap_sum[block_idx, step_idx] += metrics["subspace_overlap"]

    def finalize(self) -> dict[str, np.ndarray]:
        pair_count = self.pair_count
        sv_count = self.sv_count
        return {
            "l1_diff": _divide_or_nan(self.l1_sum, pair_count),
            "cos_sim": _divide_or_nan(self.cos_sum, pair_count),
            "grassmann_dist": _divide_or_nan(self.grassmann_sum, pair_count),
            "subspace_overlap": _divide_or_nan(self.overlap_sum, pair_count),
            "sv_spectrum": _divide_or_nan(self.sv_sum, sv_count[..., None]),
            "pair_count": pair_count.copy(),
            "sv_count": sv_count.copy(),
        }


def save_evidence_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    """Save evidence arrays and JSON metadata in one NPZ file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **arrays,
        metadata_json=np.array(json.dumps(metadata, indent=2, sort_keys=True)),
    )


def _divide_or_nan(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    return np.divide(values, counts, out=out, where=counts > 0)
