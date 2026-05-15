"""Merge host A/B c_FID result JSONs with baseline-ratio scaling.

Policy:
1) Use host B as canonical baseline/config.
2) Keep all valid B entries as-is.
3) For entries missing in B but present in A, scale A's FID by:
       scale = baseline_B / baseline_A
   then recompute delta against baseline_B.
4) Write merged JSON for subsequent A2/B2 scheduling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a-json",
        type=Path,
        default=Path("dit_s3cache/fid/fid_sensitivity_results_A.json"),
    )
    parser.add_argument(
        "--b-json",
        type=Path,
        default=Path("dit_s3cache/fid/fid_sensitivity_results_B.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("dit_s3cache/fid/fid_sensitivity_results_merged.json"),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional fixed scale. Default uses baseline_B / baseline_A.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Result JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_valid_entry(entry: Any, sample_count: int | None) -> bool:
    if not isinstance(entry, dict):
        return False
    if "fid" not in entry:
        return False
    try:
        float(entry["fid"])
    except (TypeError, ValueError):
        return False
    if sample_count is not None and entry.get("sample_count") is not None:
        if int(entry["sample_count"]) != int(sample_count):
            return False
    return True


def _iter_valid_entries(results: dict[str, Any], sample_count: int | None):
    for k_key, bucket in results.items():
        if not (isinstance(k_key, str) and k_key.startswith("k") and isinstance(bucket, dict)):
            continue
        for block_key, entry in bucket.items():
            if _is_valid_entry(entry, sample_count):
                yield k_key, block_key, entry


def merge_results(
    a_payload: dict[str, Any],
    b_payload: dict[str, Any],
    *,
    scale: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    a_results = a_payload.get("results", {})
    b_results = b_payload.get("results", {})
    if not isinstance(a_results, dict) or not isinstance(b_results, dict):
        raise ValueError("Invalid JSON structure: missing `results` dict")

    baseline_a = float(a_results["baseline_fid"])
    baseline_b = float(b_results["baseline_fid"])
    effective_scale = float(scale) if scale is not None else (baseline_b / baseline_a)

    merged = json.loads(json.dumps(b_payload))  # deep copy
    merged_results = merged.setdefault("results", {})

    sample_count = merged.get("config", {}).get("eval_samples")
    if sample_count is not None:
        sample_count = int(sample_count)

    existing = {(k, b) for k, b, _ in _iter_valid_entries(merged_results, sample_count)}

    added_from_a = 0
    skipped_conflict = 0
    for k_key, block_key, entry in _iter_valid_entries(a_results, sample_count):
        if (k_key, block_key) in existing:
            skipped_conflict += 1
            continue
        bucket = merged_results.setdefault(k_key, {})
        fid_scaled = float(entry["fid"]) * effective_scale
        bucket[block_key] = {
            "fid": fid_scaled,
            "delta": fid_scaled - baseline_b,
            "sample_count": int(sample_count) if sample_count is not None else int(entry.get("sample_count", 0)),
            "source": "A_scaled_to_B",
            "source_fid_raw": float(entry["fid"]),
            "source_delta_raw": float(entry.get("delta", 0.0)),
            "scale_applied": effective_scale,
            "cache_stats": entry.get("cache_stats", {}),
        }
        added_from_a += 1

    merged["last_updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged.setdefault("merge_info", {})
    merged["merge_info"].update(
        {
            "base_host": "B",
            "a_baseline_fid": baseline_a,
            "b_baseline_fid": baseline_b,
            "scale_applied_to_a": effective_scale,
            "added_from_a_scaled": int(added_from_a),
            "conflicts_skipped_existing_b": int(skipped_conflict),
        }
    )

    summary = {
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "scale": effective_scale,
        "added_from_a_scaled": added_from_a,
        "skipped_conflict": skipped_conflict,
    }
    return merged, summary


def main() -> None:
    args = build_parser().parse_args()
    a_payload = _load_json(args.a_json)
    b_payload = _load_json(args.b_json)
    merged, summary = merge_results(a_payload, b_payload, scale=args.scale)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)

    print(f"Merged JSON written: {args.out_json}")
    print(
        "A baseline={baseline_a:.9f}, B baseline={baseline_b:.9f}, scale={scale:.9f}".format(
            **summary
        )
    )
    print(
        "Added from A (scaled): {added_from_a_scaled}, conflicts skipped: {skipped_conflict}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()

