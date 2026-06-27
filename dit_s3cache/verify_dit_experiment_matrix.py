#!/usr/bin/env python3
"""Verify dit_experiment_matrix.json against all runs_index.jsonl files."""

import json
import sys
from pathlib import Path

DIT_S3 = Path(__file__).resolve().parent
MATRIX_PATH = DIT_S3 / "results" / "dit_experiment_matrix.json"
RESULTS_ROOT = DIT_S3 / "results"


def matrix_source_paths(matrix: dict) -> set[str]:
    paths = set()
    for exp in matrix["experiments"]:
        for key in (
            "source_file",
            "source_file_fid_50k",
            "source_file_detail_stats",
            "source_file_detail_stats_fid_50k",
        ):
            v = exp.get(key)
            if v:
                paths.add(v)
    return paths


def normalize_sum_path(sum_rel: str) -> str:
    """Turn runs_index sum path into suffix comparable to matrix absolute paths."""
    return sum_rel.replace("dit_s3cache/results/", "")


def main() -> int:
    matrix = json.loads(MATRIX_PATH.read_text())
    sources = matrix_source_paths(matrix)

    index_runs = []
    for idx_path in sorted(RESULTS_ROOT.rglob("runs_index.jsonl")):
        for line in idx_path.read_text().strip().splitlines():
            if line.strip():
                index_runs.append(json.loads(line))

    success = [r for r in index_runs if r.get("st") == "success" and r.get("sum")]
    failed = [r for r in index_runs if r.get("st") != "success"]

    missing = []
    for r in success:
        suffix = normalize_sum_path(r["sum"])
        if not any(suffix in src for src in sources):
            missing.append(r)

    mismatches = []
    for exp in matrix["experiments"]:
        for field, fid_key in [("source_file", "fid_5k"), ("source_file_fid_50k", "fid_50k")]:
            src = exp.get(field)
            if not src or not str(src).endswith("summary.json"):
                continue
            p = Path(src)
            if not p.exists():
                mismatches.append(f"{exp['id']}: missing {field}")
                continue
            s = json.loads(p.read_text())
            actual = s.get("fid_5k")
            expected = exp.get(fid_key)
            if expected is not None and actual is not None and abs(actual - expected) > 0.001:
                mismatches.append(
                    f"{exp['id']}: {fid_key} matrix={expected} summary={actual}"
                )

    print(f"Matrix: {MATRIX_PATH}")
    print(f"last_updated: {matrix['metadata'].get('last_updated')}")
    print(f"runs_index: {len(index_runs)} total, {len(success)} success, {len(failed)} failed")
    print(f"Matrix experiments: {len(matrix['experiments'])}")

    if mismatches:
        print("\nFID MISMATCHES:")
        for m in mismatches:
            print(f"  {m}")
    else:
        print("\nFID check: all matrix values match summary.json")

    curated_exclude_prefixes = (
        "baseline_p1_",
        "baseline_p3_",
        "sweep_",
        "baseline_p2_K20_sw3_lam1.0_blockwise",  # DDPM250 sweep (non-kmax3)
    )
    missing_curated = [
        r
        for r in missing
        if not any(r.get("sch", "").startswith(p) for p in curated_exclude_prefixes)
        and "sublayer" not in r.get("sum", "")
    ]

    if missing_curated:
        print("\nSUCCESS RUNS NOT IN MATRIX (may need update):")
        for r in missing_curated:
            fk = "FID@50K" if "FID@50K" in r else "FID@5K"
            print(f"  {r['rid']} {fk}={r.get(fk)} sch={r.get('sch')}")
        return 1

    if mismatches:
        return 1

    print("\nNo update needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
