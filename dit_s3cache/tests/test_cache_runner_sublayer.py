"""Unit tests for sub-layer DiT cache runner."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn as nn


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dit_s3cache.fid.cache_runner_sublayer import (  # noqa: E402
    CachedDiTBlockSubLayer,
    install_sublayer_cache_wrappers,
    make_sublayer_cached_forward_with_cfg,
    reset_sublayer_cache_state,
    restore_sublayer_cache_wrappers,
    sublayer_cache_stats,
)


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FixedAdaLN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(c.shape[0], 6 * self.dim, dtype=c.dtype, device=c.device)
        out[:, 2 * self.dim : 3 * self.dim] = 1.0
        out[:, 5 * self.dim : 6 * self.dim] = 1.0
        return out


class _Scale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _ToyDiTBlock(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.adaLN_modulation = _FixedAdaLN(dim)
        self.norm1 = _Identity()
        self.norm2 = _Identity()
        self.attn = _Scale(1.0)
        self.mlp = _Scale(2.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        from models import modulate

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        r_msa = gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + r_msa
        r_mlp = gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x + r_mlp


def _inputs(dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, dim) / 10.0
    c = torch.zeros(2, dim)
    return x, c


@pytest.mark.parametrize(
    ("msa_steps", "mlp_steps", "expected"),
    [
        ({100}, {100}, {"msa_cache_hits": 1, "mlp_cache_hits": 1}),
        ({100, 90}, {100}, {"msa_recompute_hits": 2, "mlp_cache_hits": 1}),
        ({100}, {100, 90}, {"msa_cache_hits": 1, "mlp_recompute_hits": 2}),
        ({100, 90}, {100, 90}, {"msa_recompute_hits": 2, "mlp_recompute_hits": 2}),
    ],
)
def test_four_msa_mlp_cache_combinations(
    msa_steps: set[int],
    mlp_steps: set[int],
    expected: dict[str, int],
) -> None:
    cb = CachedDiTBlockSubLayer(
        block=_ToyDiTBlock(),
        block_idx=0,
        recompute_steps_msa=msa_steps,
        recompute_steps_mlp=mlp_steps,
    )
    x, c = _inputs()

    cb.current_timestep = 100
    cb.block.forward(x, c)
    cb.current_timestep = 90
    cb.block.forward(x, c)

    stats = cb.stats()
    for key, value in expected.items():
        assert stats[key] == value
    assert stats["total_branch_calls"] == 4


def test_full_compute_matches_unpatched_forward() -> None:
    original = _ToyDiTBlock()
    patched = _ToyDiTBlock()
    cb = CachedDiTBlockSubLayer(
        block=patched,
        block_idx=0,
        recompute_steps_msa={100},
        recompute_steps_mlp={100},
    )
    x, c = _inputs()

    cb.current_timestep = 100
    out_cached_runner = cb.block.forward(x.clone(), c)
    out_original = original.forward(x.clone(), c)

    assert torch.allclose(out_cached_runner, out_original)


def test_sublayer_forward_with_cfg_broadcast_and_stats() -> None:
    model = nn.Module()
    model.blocks = nn.ModuleList([_ToyDiTBlock(), _ToyDiTBlock()])

    def forward_with_cfg(
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        c = torch.zeros(x.shape[0], x.shape[-1], dtype=x.dtype, device=x.device)
        for block in model.blocks:
            x = block(x, c)
        return x

    model.forward_with_cfg = forward_with_cfg
    scheduler = {
        (0, "msa"): {100},
        (0, "mlp"): {100},
        (1, "msa"): {100, 90},
        (1, "mlp"): {100},
    }
    cached_blocks = install_sublayer_cache_wrappers(model, scheduler)
    wrapped = make_sublayer_cached_forward_with_cfg(model, cached_blocks)
    reset_sublayer_cache_state(cached_blocks)

    x, _ = _inputs()
    y = torch.zeros(x.shape[0], dtype=torch.long)
    wrapped(x, torch.tensor([100, 100]), y, cfg_scale=1.5)
    wrapped(x, torch.tensor([90, 90]), y, cfg_scale=1.5)

    assert all(cb.current_timestep == 90 for cb in cached_blocks)
    stats = sublayer_cache_stats(cached_blocks)
    assert stats["total_branch_calls"] == 8
    assert stats["per_block"]["block_00"]["msa_cache_hits"] == 1
    assert stats["per_block"]["block_01"]["msa_recompute_hits"] == 2

    with pytest.raises(ValueError, match="uniform timestep"):
        wrapped(x, torch.tensor([80, 90]), y, cfg_scale=1.5)

    restore_sublayer_cache_wrappers(cached_blocks)
