"""Canonically sort c_FID results JSON key order for human reading.

Ordering policy:
  - Root: ``config``, ``results``, ``last_updated``, ``merge_info``, then other keys alphabetically.
  - ``results``: ``baseline_fid``, ``baseline_meta``, then ``k<number>`` keys sorted by numeric k,
    then any other keys alphabetically.
  - Each ``k*`` bucket: ``block_<i>`` sorted by numeric i.
  - Each experiment dict: ``fid``, ``delta``, ``sample_count``, remaining keys alphabetically except
    ``cache_stats`` last.
  - ``cache_stats.per_block``: numeric ``block_*`` order.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _block_index(name: str) -> int | None:
    m = re.fullmatch(r"block_(\d+)", name)
    return int(m.group(1)) if m else None


def _sort_block_keys(names: list[str]) -> list[str]:
    keyed: list[tuple[tuple[int, int], str]] = []
    for n in names:
        idx = _block_index(n)
        if idx is not None:
            keyed.append(((0, idx), n))
        else:
            keyed.append(((1, 0), n))
    keyed.sort(key=lambda t: (t[0][0], t[0][1], t[1]))
    return [n for _, n in keyed]


def _k_numeric(name: str) -> int | None:
    if name.startswith("k") and name[1:].isdigit():
        return int(name[1:])
    return None


def _sort_cache_stats(cache_stats: Any) -> Any:
    if not isinstance(cache_stats, dict):
        return cache_stats
    out: dict[str, Any] = {}
    pb = cache_stats.get("per_block")
    scalar_keys = [k for k in cache_stats.keys() if k != "per_block"]
    for k in sorted(scalar_keys):
        out[k] = cache_stats[k]
    if isinstance(pb, dict):
        ordered = _sort_block_keys(list(pb.keys()))
        out["per_block"] = {bk: pb[bk] for bk in ordered}
    elif pb is not None:
        out["per_block"] = pb
    return out


def _sort_experiment_entry(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    head = ["fid", "delta", "sample_count"]
    out: dict[str, Any] = {}
    for k in head:
        if k in entry:
            out[k] = entry[k]
    tail = [k for k in entry.keys() if k not in out and k != "cache_stats"]
    for k in sorted(tail):
        out[k] = entry[k]
    if "cache_stats" in entry:
        out["cache_stats"] = _sort_cache_stats(entry["cache_stats"])
    return out


def _sort_baseline_meta(meta: Any) -> Any:
    if not isinstance(meta, dict):
        return meta
    head = [
        "fid",
        "num_fid_samples",
        "model",
        "image_size",
        "num_sampling_steps",
        "cfg_scale",
        "seed",
        "vae",
        "gen_image_dir",
        "sample_npz",
        "ref_batch",
        "adm_evaluator",
    ]
    out: dict[str, Any] = {}
    for k in head:
        if k in meta:
            out[k] = meta[k]
    rest = [k for k in meta.keys() if k not in out and k != "cache_stats"]
    for k in sorted(rest):
        out[k] = meta[k]
    if "cache_stats" in meta:
        out["cache_stats"] = _sort_cache_stats(meta["cache_stats"])
    return out


def _sort_k_bucket(bucket: Any) -> Any:
    if not isinstance(bucket, dict):
        return bucket
    blocks = [k for k in bucket if _block_index(k) is not None]
    extra = [k for k in bucket if _block_index(k) is None]
    out: dict[str, Any] = {}
    for bk in _sort_block_keys(blocks):
        out[bk] = _sort_experiment_entry(bucket[bk])
    for k in sorted(extra):
        out[k] = bucket[k]
    return out


def _sort_results_section(results: Any) -> Any:
    if not isinstance(results, dict):
        return results
    out: dict[str, Any] = {}
    if "baseline_fid" in results:
        out["baseline_fid"] = results["baseline_fid"]
    if "baseline_meta" in results:
        out["baseline_meta"] = _sort_baseline_meta(results["baseline_meta"])
    k_names = sorted(
        (k for k in results if _k_numeric(k) is not None),
        key=lambda x: _k_numeric(x) or 0,
    )
    for kk in k_names:
        out[kk] = _sort_k_bucket(results[kk])
    used = set(out.keys())
    for k in sorted(results.keys()):
        if k not in used:
            out[k] = results[k]
    return out


def sort_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with canonical key ordering (deep where described)."""
    out: dict[str, Any] = {}
    for k in ("config", "results", "last_updated", "merge_info"):
        if k in payload:
            out[k] = payload[k]
    if "results" in out:
        out["results"] = _sort_results_section(out["results"])
    for k in sorted(payload.keys()):
        if k not in out:
            out[k] = payload[k]
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=Path("dit_s3cache/fid/fid_sensitivity_results.json"),
        help="Path to results JSON (default: dit_s3cache/fid/fid_sensitivity_results.json)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write to this path (default: overwrite input)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    path = args.json_path
    out_path = args.output or path
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("Expected JSON object at root")
    sorted_data = sort_payload(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
