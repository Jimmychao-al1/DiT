"""Cache-aware DiT sampling utilities for c_FID experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, MutableMapping

import torch

from models import modulate


CacheScheduler = MutableMapping[int, set[int]]


@dataclass
class CachedDiTBlock:
    """Monkey-patch one DiTBlock with block-residual runtime caching."""

    block: torch.nn.Module
    block_idx: int
    recompute_steps: set[int]

    def __post_init__(self) -> None:
        self.cached_residual: torch.Tensor | None = None
        self.current_timestep: int | None = None
        self.cache_hits = 0
        self.recompute_hits = 0
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

            should_recompute = (timestep in self.recompute_steps) or (self.cached_residual is None)
            if should_recompute:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    block.adaLN_modulation(c).chunk(6, dim=1)
                )
                r_msa = gate_msa.unsqueeze(1) * block.attn(
                    modulate(block.norm1(x), shift_msa, scale_msa)
                )
                x_after_msa = x + r_msa
                r_mlp = gate_mlp.unsqueeze(1) * block.mlp(
                    modulate(block.norm2(x_after_msa), shift_mlp, scale_mlp)
                )
                self.cached_residual = (r_msa + r_mlp).detach()
                self.recompute_hits += 1
                return x_after_msa + r_mlp

            self.cache_hits += 1
            return x + self.cached_residual

        self.block.forward = cached_forward
        self._is_installed = True

    def reset_cache(self) -> None:
        self.cached_residual = None
        self.current_timestep = None

    def restore(self) -> None:
        if not self._is_installed:
            return
        self.block.forward = self._original_forward
        self._is_installed = False
        self.reset_cache()

    def stats(self) -> dict[str, int]:
        return {
            "cache_hits": int(self.cache_hits),
            "recompute_hits": int(self.recompute_hits),
            "total_hook_calls": int(self.cache_hits + self.recompute_hits),
        }


def create_dit_cache_config(
    target_block: int,
    k: int,
    total_steps: int = 250,
    n_blocks: int = 28,
) -> dict[int, set[int]]:
    """Create a single-target-block recompute scheduler.

    Scheduler timesteps use diffusion timestep values, not step indices.  For
    250 steps this is ``249, 248, ..., 0``.  The first executed timestep is
    always included to avoid a cold-cache miss.
    """

    if not 0 <= target_block < n_blocks:
        raise ValueError(f"target_block must be in [0, {n_blocks}), got {target_block}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    indices = list(range(total_steps))[::-1]
    recompute_steps = {
        timestep for step_idx, timestep in enumerate(indices) if step_idx % k == 0
    }
    recompute_steps.add(indices[0])

    return {
        block_idx: (set(range(total_steps)) if block_idx != target_block else recompute_steps)
        for block_idx in range(n_blocks)
    }


def install_cache_wrappers(
    model: torch.nn.Module,
    cache_scheduler: CacheScheduler,
) -> list[CachedDiTBlock]:
    """Patch all model blocks according to a cache scheduler."""

    cached_blocks: list[CachedDiTBlock] = []
    for block_idx, block in enumerate(model.blocks):
        recompute_steps = cache_scheduler.get(block_idx, set())
        cached_blocks.append(
            CachedDiTBlock(
                block=block,
                block_idx=block_idx,
                recompute_steps=set(int(step) for step in recompute_steps),
            )
        )
    return cached_blocks


def restore_cache_wrappers(cached_blocks: Iterable[CachedDiTBlock]) -> None:
    for cached_block in cached_blocks:
        cached_block.restore()


def reset_cache_state(cached_blocks: Iterable[CachedDiTBlock]) -> None:
    for cached_block in cached_blocks:
        cached_block.reset_cache()


def make_cached_forward_with_cfg(
    model: torch.nn.Module,
    cached_blocks: list[CachedDiTBlock],
) -> Callable[..., torch.Tensor]:
    """Wrap ``model.forward_with_cfg`` and broadcast timestep to cached blocks."""

    original_forward_with_cfg = model.forward_with_cfg

    def wrapped(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        current_timestep = int(t[0].item())
        for cached_block in cached_blocks:
            cached_block.current_timestep = current_timestep
        return original_forward_with_cfg(x, t, y, cfg_scale)

    return wrapped


def cache_stats(cached_blocks: Iterable[CachedDiTBlock]) -> dict[str, int | dict[str, int]]:
    per_block = {f"block_{cb.block_idx}": cb.stats() for cb in cached_blocks}
    total_cache_hits = sum(cb.cache_hits for cb in cached_blocks)
    total_recompute_hits = sum(cb.recompute_hits for cb in cached_blocks)
    return {
        "cache_hits": int(total_cache_hits),
        "recompute_hits": int(total_recompute_hits),
        "total_hook_calls": int(total_cache_hits + total_recompute_hits),
        "per_block": per_block,
    }
