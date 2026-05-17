"""Sub-layer cache-aware DiT sampling utilities.

This is the core runtime wrapper for 56 cacheable units.  It is deliberately
not wired into the FID sampling CLI yet; Stage 1 still needs to emit a
sub-layer scheduler first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import torch

from models import modulate


BRANCHES: tuple[str, str] = ("msa", "mlp")
SubLayerKey = tuple[int, str]
SubLayerCacheScheduler = Mapping[SubLayerKey, set[int]]


@dataclass
class CachedDiTBlockSubLayer:
    """Monkey-patch one DiTBlock with separate MSA/MLP residual caches."""

    block: torch.nn.Module
    block_idx: int
    recompute_steps_msa: set[int]
    recompute_steps_mlp: set[int]

    def __post_init__(self) -> None:
        self.cached_msa: torch.Tensor | None = None
        self.cached_mlp: torch.Tensor | None = None
        self.current_timestep: int | None = None
        self.msa_cache_hits = 0
        self.msa_recompute_hits = 0
        self.mlp_cache_hits = 0
        self.mlp_recompute_hits = 0
        self._original_forward: Callable[..., torch.Tensor] = self.block.forward
        self._is_installed = False
        self.install()

    def install(self) -> None:
        if self._is_installed:
            return

        block = self.block

        def cached_forward(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            timestep = self.current_timestep
            if timestep is None:
                raise RuntimeError(f"Block {self.block_idx}: current_timestep is not set")

            compute_msa = (timestep in self.recompute_steps_msa) or (self.cached_msa is None)
            compute_mlp = (timestep in self.recompute_steps_mlp) or (self.cached_mlp is None)

            if compute_msa or compute_mlp:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    block.adaLN_modulation(c).chunk(6, dim=1)
                )
            else:
                shift_msa = scale_msa = gate_msa = None
                shift_mlp = scale_mlp = gate_mlp = None

            if compute_msa:
                assert shift_msa is not None and scale_msa is not None and gate_msa is not None
                r_msa = gate_msa.unsqueeze(1) * block.attn(
                    modulate(block.norm1(x), shift_msa, scale_msa)
                )
                self.cached_msa = r_msa.detach()
                self.msa_recompute_hits += 1
            else:
                r_msa = self.cached_msa
                self.msa_cache_hits += 1

            if r_msa is None:
                raise RuntimeError(f"Block {self.block_idx}: MSA cache is unexpectedly empty")
            x_after_msa = x + r_msa

            if compute_mlp:
                assert shift_mlp is not None and scale_mlp is not None and gate_mlp is not None
                r_mlp = gate_mlp.unsqueeze(1) * block.mlp(
                    modulate(block.norm2(x_after_msa), shift_mlp, scale_mlp)
                )
                self.cached_mlp = r_mlp.detach()
                self.mlp_recompute_hits += 1
            else:
                r_mlp = self.cached_mlp
                self.mlp_cache_hits += 1

            if r_mlp is None:
                raise RuntimeError(f"Block {self.block_idx}: MLP cache is unexpectedly empty")
            return x_after_msa + r_mlp

        self.block.forward = cached_forward
        self._is_installed = True

    def reset_cache(self) -> None:
        self.cached_msa = None
        self.cached_mlp = None
        self.current_timestep = None

    def restore(self) -> None:
        if not self._is_installed:
            return
        self.block.forward = self._original_forward
        self._is_installed = False
        self.reset_cache()

    def stats(self) -> dict[str, int]:
        total_cache_hits = self.msa_cache_hits + self.mlp_cache_hits
        total_recompute_hits = self.msa_recompute_hits + self.mlp_recompute_hits
        return {
            "msa_cache_hits": int(self.msa_cache_hits),
            "msa_recompute_hits": int(self.msa_recompute_hits),
            "mlp_cache_hits": int(self.mlp_cache_hits),
            "mlp_recompute_hits": int(self.mlp_recompute_hits),
            "cache_hits": int(total_cache_hits),
            "recompute_hits": int(total_recompute_hits),
            "total_branch_calls": int(total_cache_hits + total_recompute_hits),
        }


def install_sublayer_cache_wrappers(
    model: torch.nn.Module,
    cache_scheduler: SubLayerCacheScheduler,
) -> list[CachedDiTBlockSubLayer]:
    """Patch all model blocks according to a sub-layer cache scheduler."""

    cached_blocks: list[CachedDiTBlockSubLayer] = []
    for block_idx, block in enumerate(model.blocks):
        cached_blocks.append(
            CachedDiTBlockSubLayer(
                block=block,
                block_idx=block_idx,
                recompute_steps_msa=set(int(step) for step in cache_scheduler.get((block_idx, "msa"), set())),
                recompute_steps_mlp=set(int(step) for step in cache_scheduler.get((block_idx, "mlp"), set())),
            )
        )
    return cached_blocks


def restore_sublayer_cache_wrappers(cached_blocks: Iterable[CachedDiTBlockSubLayer]) -> None:
    for cached_block in cached_blocks:
        cached_block.restore()


def reset_sublayer_cache_state(cached_blocks: Iterable[CachedDiTBlockSubLayer]) -> None:
    for cached_block in cached_blocks:
        cached_block.reset_cache()


def make_sublayer_cached_forward_with_cfg(
    model: torch.nn.Module,
    cached_blocks: list[CachedDiTBlockSubLayer],
) -> Callable[..., torch.Tensor]:
    """Wrap ``model.forward_with_cfg`` and broadcast timestep to cached blocks."""

    original_forward_with_cfg = model.forward_with_cfg

    def wrapped(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        if not torch.all(t == t[0]):
            raise ValueError(
                f"Expected uniform timestep across batch, got: {t.tolist()}"
            )
        current_timestep = int(t[0].item())
        for cached_block in cached_blocks:
            cached_block.current_timestep = current_timestep
        return original_forward_with_cfg(x, t, y, cfg_scale)

    return wrapped


def sublayer_cache_stats(
    cached_blocks: Iterable[CachedDiTBlockSubLayer],
) -> dict[str, int | dict[str, dict[str, int]]]:
    per_block = {f"block_{cb.block_idx:02d}": cb.stats() for cb in cached_blocks}
    total_msa_cache_hits = sum(cb.msa_cache_hits for cb in cached_blocks)
    total_msa_recompute_hits = sum(cb.msa_recompute_hits for cb in cached_blocks)
    total_mlp_cache_hits = sum(cb.mlp_cache_hits for cb in cached_blocks)
    total_mlp_recompute_hits = sum(cb.mlp_recompute_hits for cb in cached_blocks)
    total_cache_hits = total_msa_cache_hits + total_mlp_cache_hits
    total_recompute_hits = total_msa_recompute_hits + total_mlp_recompute_hits
    return {
        "msa_cache_hits": int(total_msa_cache_hits),
        "msa_recompute_hits": int(total_msa_recompute_hits),
        "mlp_cache_hits": int(total_mlp_cache_hits),
        "mlp_recompute_hits": int(total_mlp_recompute_hits),
        "cache_hits": int(total_cache_hits),
        "recompute_hits": int(total_recompute_hits),
        "total_branch_calls": int(total_cache_hits + total_recompute_hits),
        "per_block": per_block,
    }
