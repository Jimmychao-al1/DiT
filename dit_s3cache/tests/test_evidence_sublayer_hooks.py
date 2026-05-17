"""Tests for DiT sub-layer evidence hooks."""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dit_s3cache.evidence.hooks_sublayer import (  # noqa: E402
    install_dit_sublayer_hooks,
    restore_sublayer_hooks,
    sublayer_name,
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


def test_sublayer_hook_stores_msa_and_mlp_separately() -> None:
    model = nn.Module()
    model.blocks = nn.ModuleList([_ToyDiTBlock(), _ToyDiTBlock()])
    storage: dict[int, dict[str, torch.Tensor]] = {}
    hooks = install_dit_sublayer_hooks(model, storage)

    try:
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0
        c = torch.zeros(2, 4)
        expected = x
        for block in model.blocks:
            expected = block._original_forward(expected, c) if hasattr(block, "_original_forward") else block.forward(expected, c)

        out = x
        for block in model.blocks:
            out = block(out, c)

        assert torch.allclose(out, expected)
        assert set(storage.keys()) == {0, 1}
        for block_idx in range(2):
            assert set(storage[block_idx].keys()) == {"msa", "mlp"}
            assert storage[block_idx]["msa"].shape == x.shape
            assert storage[block_idx]["mlp"].shape == x.shape
            assert not torch.allclose(storage[block_idx]["msa"], storage[block_idx]["mlp"])
            assert model.blocks[block_idx]._cached_r_msa is storage[block_idx]["msa"]
            assert model.blocks[block_idx]._cached_r_mlp is storage[block_idx]["mlp"]
    finally:
        restore_sublayer_hooks(hooks)


def test_sublayer_name_is_zero_padded() -> None:
    assert sublayer_name(0, "msa") == "block_00_msa"
    assert sublayer_name(27, "mlp") == "block_27_mlp"
