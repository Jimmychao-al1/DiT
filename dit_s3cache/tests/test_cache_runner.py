"""Unit tests for dit_s3cache.fid.cache_runner.

Tests cover:
  - create_dit_cache_config: correct target vs non-target recompute sets
  - CachedDiTBlock: recompute / cache hit counting
  - make_cached_forward_with_cfg: timestep broadcast, uniform-t guard
  - End-to-end mini denoising loop with a toy DiTBlock

Run from the DiT repo root:
    PYTHONPATH=/path/to/DiT python -m pytest dit_s3cache/tests/
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path bootstrap so we can import cache_runner without installing the package
# ---------------------------------------------------------------------------
import sys
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # DiT/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dit_s3cache.fid.cache_runner import (
    CachedDiTBlock,
    cache_stats,
    create_dit_cache_config,
    install_cache_wrappers,
    make_cached_forward_with_cfg,
    reset_cache_state,
    restore_cache_wrappers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_sampling_timesteps(n: int = 10) -> list[int]:
    """Simulate n reversed SpacedDiffusion timestep_map values (e.g. 999,995,…)."""
    step = 1000 // n
    return [1000 - step * (i + 1) for i in range(n)]


class _FakeDiTBlock(nn.Module):
    """Minimal drop-in for DiTBlock with the same attribute API.

    All sub-modules are simple linear / identity layers so the tests run
    without a real DiT checkpoint or GPU.

    Layout:
      adaLN_modulation(c) → (N, 6*D)  [6 chunks used by CachedDiTBlock]
      attn(x)             → (N, T, D)  [identity-like]
      mlp(x)              → (N, T, D)  [identity-like]
      norm1/norm2         → LayerNorm
    """

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.adaLN_modulation = nn.Sequential(nn.Linear(dim, 6 * dim, bias=False))
        nn.init.zeros_(self.adaLN_modulation[0].weight)   # deterministic zeros out
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        from models import modulate
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        # x shape: (N, T, D); c shape: (N, D)
        r_msa = gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + r_msa
        r_mlp = gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x + r_mlp


def _make_model_with_blocks(n_blocks: int = 4, dim: int = 8) -> nn.Module:
    """Build a toy model whose forward_with_cfg drives all blocks."""
    model = nn.Module()
    model.blocks = nn.ModuleList([_FakeDiTBlock(dim) for _ in range(n_blocks)])

    def forward_with_cfg(
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        # x: (N, T, D),  c: (N, D)
        c = t.float().unsqueeze(-1).expand(-1, x.shape[-1])
        for block in model.blocks:
            x = block(x, c)
        return x

    model.forward_with_cfg = forward_with_cfg
    return model


# ---------------------------------------------------------------------------
# Tests: create_dit_cache_config
# ---------------------------------------------------------------------------

class TestCreateDitCacheConfig:
    def test_all_blocks_present(self):
        ts = _fake_sampling_timesteps(10)
        cfg = create_dit_cache_config(target_block=2, k=3, sampling_timesteps=ts, n_blocks=5)
        assert set(cfg.keys()) == {0, 1, 2, 3, 4}

    def test_non_target_blocks_get_all_steps(self):
        ts = _fake_sampling_timesteps(10)
        cfg = create_dit_cache_config(target_block=2, k=3, sampling_timesteps=ts, n_blocks=5)
        all_steps = set(ts)
        for block_idx in range(5):
            if block_idx != 2:
                assert cfg[block_idx] == all_steps, (
                    f"block {block_idx} should recompute every step"
                )

    def test_target_block_recompute_cadence(self):
        ts = list(range(100, 0, -10))  # [100,90,80,...,10]
        k = 3
        cfg = create_dit_cache_config(target_block=1, k=k, sampling_timesteps=ts, n_blocks=3)
        target_set = cfg[1]
        # step indices 0,3,6,9 → ts[0]=100, ts[3]=70, ts[6]=40, ts[9]=10
        expected = {ts[i] for i in range(len(ts)) if i % k == 0}
        expected.add(ts[0])  # first step always included
        assert target_set == expected

    def test_first_step_always_recomputed(self):
        ts = _fake_sampling_timesteps(10)
        cfg = create_dit_cache_config(target_block=0, k=7, sampling_timesteps=ts, n_blocks=2)
        assert ts[0] in cfg[0], "First denoising step must always be in target recompute set"

    def test_target_recompute_subset_of_all(self):
        ts = _fake_sampling_timesteps(20)
        cfg = create_dit_cache_config(target_block=0, k=5, sampling_timesteps=ts, n_blocks=2)
        assert cfg[0].issubset(set(ts))

    def test_invalid_target_block(self):
        ts = _fake_sampling_timesteps(5)
        with pytest.raises(ValueError, match="target_block"):
            create_dit_cache_config(target_block=10, k=2, sampling_timesteps=ts, n_blocks=5)

    def test_invalid_k(self):
        ts = _fake_sampling_timesteps(5)
        with pytest.raises(ValueError, match="k must be positive"):
            create_dit_cache_config(target_block=0, k=0, sampling_timesteps=ts, n_blocks=3)

    def test_empty_sampling_timesteps(self):
        with pytest.raises(ValueError, match="empty"):
            create_dit_cache_config(target_block=0, k=1, sampling_timesteps=[], n_blocks=2)


# ---------------------------------------------------------------------------
# Tests: CachedDiTBlock
# ---------------------------------------------------------------------------

class TestCachedDiTBlock:
    DIM = 8

    def _make_cb(self, recompute_steps: set[int]) -> CachedDiTBlock:
        return CachedDiTBlock(
            block=_FakeDiTBlock(self.DIM),
            block_idx=0,
            recompute_steps=recompute_steps,
        )

    def _inputs(self):
        # x: (batch=2, tokens=4, dim=8), c: (batch=2, dim=8)
        x = torch.zeros(2, 4, self.DIM)
        c = torch.zeros(2, self.DIM)
        return x, c

    def test_recompute_on_recompute_step(self):
        cb = self._make_cb(recompute_steps={999})
        cb.current_timestep = 999
        x, c = self._inputs()
        cb.block.forward(x, c)
        assert cb.recompute_hits == 1
        assert cb.cache_hits == 0

    def test_cache_hit_on_non_recompute_step(self):
        cb = self._make_cb(recompute_steps={999})
        x, c = self._inputs()
        cb.current_timestep = 999
        cb.block.forward(x, c)              # recompute → fills cache
        cb.current_timestep = 995           # not in recompute set
        cb.block.forward(x, c)              # should use cache
        assert cb.recompute_hits == 1
        assert cb.cache_hits == 1

    def test_first_call_always_recomputes_even_if_not_in_set(self):
        """Cold cache (no cached_residual) must force recompute."""
        cb = self._make_cb(recompute_steps=set())
        cb.current_timestep = 42
        x, c = self._inputs()
        cb.block.forward(x, c)
        assert cb.recompute_hits == 1, "Cold cache must force recompute"

    def test_reset_clears_cache(self):
        cb = self._make_cb(recompute_steps={100})
        x, c = self._inputs()
        cb.current_timestep = 100
        cb.block.forward(x, c)
        cb.reset_cache()
        assert cb.cached_residual is None
        assert cb.current_timestep is None

    def test_restore_removes_patch(self):
        fake = _FakeDiTBlock(self.DIM)
        original_func = fake.forward.__func__  # underlying unbound function
        cb = CachedDiTBlock(block=fake, block_idx=0, recompute_steps={0})
        # After patch, forward is the cached_forward closure (not a bound method)
        assert fake.forward is not cb._original_forward
        cb.restore()
        assert not cb._is_installed
        # After restore, the instance attribute is removed → class method is used again
        assert fake.forward.__func__ is original_func


# ---------------------------------------------------------------------------
# Tests: make_cached_forward_with_cfg
# ---------------------------------------------------------------------------

class TestMakeCachedForwardWithCfg:
    DIM = 8

    def _setup(self, n_blocks: int = 3):
        model = _make_model_with_blocks(n_blocks, self.DIM)
        ts = list(range(100, 0, -10))
        cfg = create_dit_cache_config(
            target_block=1,
            k=3,
            sampling_timesteps=ts,
            n_blocks=n_blocks,
        )
        cache_blocks = install_cache_wrappers(model, cfg)
        wrapped = make_cached_forward_with_cfg(model, cache_blocks)
        return model, cache_blocks, wrapped

    def test_timestep_is_broadcast_to_all_blocks(self):
        _, cache_blocks, wrapped = self._setup()
        # x: (batch=4, tokens=4, dim=8)
        x = torch.zeros(4, 4, self.DIM)
        t = torch.tensor([90, 90, 90, 90])
        y = torch.zeros(4, dtype=torch.long)
        wrapped(x, t, y, cfg_scale=1.5)
        for cb in cache_blocks:
            assert cb.current_timestep == 90

    def test_non_uniform_t_raises(self):
        _, _, wrapped = self._setup()
        x = torch.zeros(4, 4, self.DIM)
        t = torch.tensor([90, 90, 80, 90])  # non-uniform
        y = torch.zeros(4, dtype=torch.long)
        with pytest.raises(ValueError, match="uniform timestep"):
            wrapped(x, t, y, cfg_scale=1.5)


# ---------------------------------------------------------------------------
# End-to-end: mini denoising loop
# ---------------------------------------------------------------------------

class TestEndToEndDenoisingLoop:
    """Simulate N denoising steps and verify recompute vs cache counts."""

    DIM = 8

    def _run_loop(self, n_blocks: int, ts: list[int], k: int, target: int):
        model = _make_model_with_blocks(n_blocks, self.DIM)
        cfg = create_dit_cache_config(
            target_block=target, k=k, sampling_timesteps=ts, n_blocks=n_blocks
        )
        cache_blocks = install_cache_wrappers(model, cfg)
        wrapped = make_cached_forward_with_cfg(model, cache_blocks)
        reset_cache_state(cache_blocks)

        # x: (batch=2, tokens=4, dim)
        x = torch.zeros(2, 4, self.DIM)
        for t_val in ts:
            t_tensor = torch.tensor([t_val, t_val])
            y = torch.zeros(2, dtype=torch.long)
            wrapped(x, t_tensor, y, cfg_scale=1.5)

        return cache_blocks

    def test_non_target_block_always_recomputes(self):
        ts = list(range(100, 0, -10))  # 10 steps
        n_steps = len(ts)
        target = 1
        cache_blocks = self._run_loop(n_blocks=3, ts=ts, k=3, target=target)

        for cb in cache_blocks:
            total = cb.recompute_hits + cb.cache_hits
            assert total == n_steps, (
                f"block {cb.block_idx}: expected {n_steps} calls, got {total}"
            )
            if cb.block_idx != target:
                assert cb.recompute_hits == n_steps, (
                    f"Non-target block {cb.block_idx} should recompute every step; "
                    f"recompute={cb.recompute_hits}, cache={cb.cache_hits}"
                )

    def test_target_block_recompute_cadence(self):
        ts = list(range(100, 0, -10))  # 10 steps
        k = 3
        target = 0
        cache_blocks = self._run_loop(n_blocks=3, ts=ts, k=k, target=target)

        expected_recomputes = len({t for i, t in enumerate(ts) if i % k == 0})
        target_cb = cache_blocks[target]
        assert target_cb.recompute_hits == expected_recomputes, (
            f"recompute_hits={target_cb.recompute_hits}, expected={expected_recomputes}"
        )
        expected_cache = len(ts) - expected_recomputes
        assert target_cb.cache_hits == expected_cache, (
            f"cache_hits={target_cb.cache_hits}, expected={expected_cache}"
        )

    def test_cache_reuse_correctness(self):
        """Cache path (x + cached_residual) must equal recompute when inputs are identical."""
        fake = _FakeDiTBlock(self.DIM)
        cb = CachedDiTBlock(block=fake, block_idx=0, recompute_steps={100})

        # x: (N, T, D),  c: (N, D)
        x = torch.rand(2, 4, self.DIM)
        c = torch.zeros(2, self.DIM)  # zeros → adaLN_modulation outputs zeros → gates=0

        # Step 1: recompute (timestep in recompute_steps)
        cb.current_timestep = 100
        out_recompute = cb.block.forward(x.clone(), c)

        # Step 2: cache hit (same x, same residual stored)
        cb.current_timestep = 90
        out_cache = cb.block.forward(x.clone(), c)

        # When gates=0 (adaLN_modulation is zero-init), residual = 0 and output = x.
        # Both paths must return the same value.
        assert torch.allclose(out_recompute, out_cache, atol=1e-6), (
            "Cache path and recompute path disagree on identical inputs"
        )
