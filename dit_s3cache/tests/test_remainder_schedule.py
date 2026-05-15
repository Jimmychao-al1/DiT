"""Tests for A2/B2 remainder scheduling (legacy weighted lists + JSON skip)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dit_s3cache.fid.fid_cache_sensitivity import (
    _legacy_remainder_pending,
    get_task_list,
    legacy_weighted_part_a_tasks,
    legacy_weighted_part_b_tasks,
    should_skip,
)


def test_part_a_31_tasks_default_n_k5_3():
    n = 28
    a = legacy_weighted_part_a_tasks(n, 3)
    assert len(a) == 31


def test_part_b_53_tasks_with_n_k5_3():
    n = 28
    b = legacy_weighted_part_b_tasks(n, 3)
    assert len(b) == 53


def test_pending_from_json_not_tail_assumption():
    n = 28
    n_k5 = 3
    results = {"results": {}}
    args = argparse.Namespace(num_fid_samples=1000)
    _a, _b, pending = _legacy_remainder_pending(n, n_k5, results, args)
    assert len(pending) == 31 + 53


def test_a2_b2_split_empty_json():
    n = 28
    n_k5 = 3
    empty = {"results": {}}
    args = argparse.Namespace(num_fid_samples=1000)
    a2 = get_task_list("A2", [3, 5, 10], n, n_k5_on_part_a_override=n_k5, results=empty, args=args)
    b2 = get_task_list("B2", [3, 5, 10], n, n_k5_on_part_a_override=n_k5, results=empty, args=args)
    assert len(a2) + len(b2) == 84
    assert not (set(a2) & set(b2))


def test_b_side_tail_after_32_in_order_matches_k10_from_7():
    n = 28
    n_k5 = 3
    results: dict = {"results": {"k5": {}, "k10": {}}}
    for b in range(n_k5, n):
        results["results"]["k5"][f"block_{b}"] = {"fid": 1.0, "sample_count": 1000}
    for b in range(7):
        results["results"]["k10"][f"block_{b}"] = {"fid": 1.0, "sample_count": 1000}
    args = argparse.Namespace(num_fid_samples=1000)
    _a, _b, pending = _legacy_remainder_pending(n, n_k5, results, args)
    pending_b = [t for t in pending if t[0] == 10]
    assert pending_b == [(10, j) for j in range(7, n)]


def test_should_skip_requires_fid():
    args = argparse.Namespace(num_fid_samples=1000)
    results = {"results": {"k3": {"block_0": {"partial": True}}}}
    assert should_skip(results, args, 3, 0) is False
