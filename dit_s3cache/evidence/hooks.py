"""Hooks for collecting DiTBlock residual outputs.

The original DiTBlock implementation keeps the attention and MLP residuals
inline.  For Stage 0 evidence collection we monkey-patch each block forward so
the model output stays identical while exposing ``r_msa + r_mlp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, MutableMapping

import torch

from models import modulate


ResidualStorage = MutableMapping[int, torch.Tensor]


@dataclass
class DiTBlockHook:
    """Monkey-patch one DiTBlock and store its block residual each forward."""

    block: torch.nn.Module
    block_idx: int
    storage: ResidualStorage

    def __post_init__(self) -> None:
        self._original_forward: Callable[..., torch.Tensor] = self.block.forward
        self._is_installed = False
        self.install()

    def install(self) -> None:
        if self._is_installed:
            return

        block = self.block
        block_idx = self.block_idx
        storage = self.storage

        def hooked_forward(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                block.adaLN_modulation(c).chunk(6, dim=1)
            )
            r_msa = gate_msa.unsqueeze(1) * block.attn(
                modulate(block.norm1(x), shift_msa, scale_msa)
            )
            x = x + r_msa
            r_mlp = gate_mlp.unsqueeze(1) * block.mlp(
                modulate(block.norm2(x), shift_mlp, scale_mlp)
            )
            x = x + r_mlp

            storage[block_idx] = (r_msa + r_mlp).detach()
            return x

        block.forward = hooked_forward
        self._is_installed = True

    def restore(self) -> None:
        if not self._is_installed:
            return
        self.block.forward = self._original_forward
        self._is_installed = False

    def get_residual(self) -> torch.Tensor | None:
        return self.storage.get(self.block_idx)

    def clear(self) -> None:
        self.storage.pop(self.block_idx, None)


def install_dit_block_hooks(
    model: torch.nn.Module, storage: ResidualStorage
) -> list[DiTBlockHook]:
    """Install hooks on all blocks in a DiT model."""

    return [
        DiTBlockHook(block=block, block_idx=block_idx, storage=storage)
        for block_idx, block in enumerate(model.blocks)
    ]


def restore_hooks(hooks: list[DiTBlockHook]) -> None:
    """Restore all patched block forwards."""

    for hook in hooks:
        hook.restore()
