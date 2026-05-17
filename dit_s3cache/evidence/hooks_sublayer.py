"""Sub-layer hooks for collecting DiTBlock MSA and MLP residual branches.

This module is intentionally parallel to ``hooks.py``.  The block-level path
stores ``r_msa + r_mlp``; this path stores the gated branches separately so
evidence can be computed for 56 cacheable units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, MutableMapping

import torch

from models import modulate


BRANCHES: tuple[str, str] = ("msa", "mlp")
SubLayerStorage = MutableMapping[int, dict[str, torch.Tensor]]


def sublayer_name(block_idx: int, branch: str) -> str:
    if branch not in BRANCHES:
        raise ValueError(f"Unknown DiT sub-layer branch: {branch!r}")
    return f"block_{block_idx:02d}_{branch}"


@dataclass
class DiTBlockSubLayerHook:
    """Monkey-patch one DiTBlock and expose gated MSA/MLP residual branches."""

    block: torch.nn.Module
    block_idx: int
    storage: SubLayerStorage

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

            r_msa_detached = r_msa.detach()
            r_mlp_detached = r_mlp.detach()
            block._cached_r_msa = r_msa_detached
            block._cached_r_mlp = r_mlp_detached
            storage[block_idx] = {
                "msa": r_msa_detached,
                "mlp": r_mlp_detached,
            }
            return x

        block.forward = hooked_forward
        self._is_installed = True

    def restore(self) -> None:
        if not self._is_installed:
            return
        self.block.forward = self._original_forward
        self._is_installed = False

    def get_residual(self, branch: str) -> torch.Tensor | None:
        if branch not in BRANCHES:
            raise ValueError(f"Unknown DiT sub-layer branch: {branch!r}")
        return self.storage.get(self.block_idx, {}).get(branch)

    def clear(self) -> None:
        self.storage.pop(self.block_idx, None)


def install_dit_sublayer_hooks(
    model: torch.nn.Module, storage: SubLayerStorage
) -> list[DiTBlockSubLayerHook]:
    """Install sub-layer hooks on all blocks in a DiT model."""

    return [
        DiTBlockSubLayerHook(block=block, block_idx=block_idx, storage=storage)
        for block_idx, block in enumerate(model.blocks)
    ]


def restore_sublayer_hooks(hooks: list[DiTBlockSubLayerHook]) -> None:
    """Restore all patched block forwards."""

    for hook in hooks:
        hook.restore()
