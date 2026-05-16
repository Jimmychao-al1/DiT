"""Unit tests for stage2 JSON → DiT cache_scheduler (raw_t) mapping."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class TestStage2ToDitCache(unittest.TestCase):
    def test_roundtrip_step_index_via_ddpm_formula(self):
        """expanded_mask True at step_idx i → raw_t = st[i]."""
        from dit_s3cache.start_run.stage2_to_dit_cache import stage2_json_to_dit_cache_scheduler
        from dit_s3cache.stage2.stage2_scheduler_adapter_dit import step_index_to_ddpm_t

        T = 10
        st = [500 + i for i in range(T)]
        row = [i % 2 == 0 for i in range(T)]
        zones = [{"id": 0, "t_start": T - 1, "t_end": 0, "length": T}]
        blocks = [
            {
                "id": bi,
                "name": f"block_{bi}",
                "expanded_mask": row if bi == 0 else [True] * T,
                "k_per_zone": [1],
            }
            for bi in range(28)
        ]
        cfg = {
            "time_order": "ddpm_249_to_0",
            "T": T,
            "shared_zones": zones,
            "blocks": blocks,
        }
        sched = stage2_json_to_dit_cache_scheduler(cfg, st, validate_cfg=False)
        for si, full in enumerate(row):
            _ = step_index_to_ddpm_t(si, T)
            rt = st[si]
            if full:
                self.assertIn(rt, sched[0])
            else:
                self.assertNotIn(rt, sched[0])

    def test_full_compute_matches_all_steps(self):
        from dit_s3cache.start_run.stage2_to_dit_cache import (
            EXPECTED_NUM_BLOCKS,
            full_compute_dit_cache_scheduler,
        )

        st = list(range(200, 200 + 28))
        self.assertEqual(len(st), len(set(st)))
        full = full_compute_dit_cache_scheduler(st, require_injective_timesteps=True)
        self.assertEqual(len(full), EXPECTED_NUM_BLOCKS)
        base = set(st)
        for bid in range(EXPECTED_NUM_BLOCKS):
            self.assertEqual(full[bid], base)

    def test_real_p2_scheduler_recompute_count_matches_mask(self):
        """If P2 Stage2 output exists: 28 keys, |recompute| == sum(expanded_mask)."""
        from dit_s3cache.start_run.stage2_to_dit_cache import (
            EXPECTED_NUM_BLOCKS,
            stage2_json_to_dit_cache_scheduler,
        )

        repo = Path(__file__).resolve().parents[2]
        sched_path = repo / (
            "dit_s3cache/stage2/stage2_output/src_baseline_p2_K20_sw3_lam1.0/"
            "02_refined_blockwise/stage2_refined_scheduler_config.json"
        )
        if not sched_path.is_file():
            self.skipTest(f"scheduler not found: {sched_path}")

        cfg = json.loads(sched_path.read_text(encoding="utf-8"))
        T = int(cfg["T"])
        timestep_map_like = list(range(999, 999 - T * 3, -3))[:T]
        self.assertEqual(len(timestep_map_like), T)
        self.assertEqual(len(timestep_map_like), len(set(timestep_map_like)))

        sched = stage2_json_to_dit_cache_scheduler(cfg, timestep_map_like, validate_cfg=True)
        self.assertEqual(set(sched.keys()), set(range(EXPECTED_NUM_BLOCKS)))

        blocks = sorted(cfg["blocks"], key=lambda b: int(b["id"]))
        for b in blocks:
            bid = int(b["id"])
            n_true = sum(1 for v in b["expanded_mask"] if v)
            self.assertEqual(len(sched[bid]), n_true)


if __name__ == "__main__":
    unittest.main()
