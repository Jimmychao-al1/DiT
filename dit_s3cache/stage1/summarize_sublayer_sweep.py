"""Summarize DiT sub-layer Stage 1 sweep outputs into a comparison CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("dit_s3cache/stage1/stage1_output_sublayer")
DEFAULT_OUTPUT_CSV = DEFAULT_INPUT_DIR / "stage1_sublayer_comparison.csv"
CSV_FIELDS = [
    "K",
    "sw",
    "lambda",
    "full_compute_ratio",
    "msa_mean_compute_ratio",
    "mlp_mean_compute_ratio",
    "msa_mlp_gap",
    "total_compute_cells",
    "total_cells",
]


def collect_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(input_dir.glob("sweep_K*_sw*_lam*/sweep_summary.json")):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        meta = _parse_sweep_dir(summary_path.parent.name)
        K = int(summary.get("K", meta["K"]))
        sw = int(summary.get("sw", meta["sw"]))
        lam = float(summary.get("lambda", meta["lambda"]))
        msa = float(summary["msa_mean_compute_ratio"])
        mlp = float(summary["mlp_mean_compute_ratio"])
        rows.append(
            {
                "K": K,
                "sw": sw,
                "lambda": lam,
                "full_compute_ratio": float(summary["full_compute_ratio"]),
                "msa_mean_compute_ratio": msa,
                "mlp_mean_compute_ratio": mlp,
                "msa_mlp_gap": msa - mlp,
                "total_compute_cells": int(summary["total_compute_cells"]),
                "total_cells": int(summary["total_cells"]),
            }
        )

    rows.sort(key=lambda r: float(r["full_compute_ratio"]))
    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"

    formatted = []
    for row in rows:
        formatted.append(
            {
                "K": f"{int(row['K'])}",
                "sw": f"{int(row['sw'])}",
                "lambda": f"{float(row['lambda']):.1f}",
                "full_compute_ratio": f"{float(row['full_compute_ratio']):.6f}",
                "msa_mean_compute_ratio": f"{float(row['msa_mean_compute_ratio']):.6f}",
                "mlp_mean_compute_ratio": f"{float(row['mlp_mean_compute_ratio']):.6f}",
                "msa_mlp_gap": f"{float(row['msa_mlp_gap']):.6f}",
                "total_compute_cells": f"{int(row['total_compute_cells'])}",
                "total_cells": f"{int(row['total_cells'])}",
            }
        )

    widths = {
        field: max(len(field), *(len(row[field]) for row in formatted))
        for field in CSV_FIELDS
    }
    header = "  ".join(field.ljust(widths[field]) for field in CSV_FIELDS)
    sep = "  ".join("-" * widths[field] for field in CSV_FIELDS)
    body = [
        "  ".join(row[field].rjust(widths[field]) for field in CSV_FIELDS)
        for row in formatted
    ]
    return "\n".join([header, sep, *body])


def _parse_sweep_dir(name: str) -> dict[str, float | int]:
    m = re.match(r"^sweep_K(?P<K>\d+)_sw(?P<sw>\d+)_lam(?P<lam>[0-9.]+)$", name)
    if not m:
        raise ValueError(f"unrecognized sweep directory name: {name!r}")
    return {
        "K": int(m.group("K")),
        "sw": int(m.group("sw")),
        "lambda": float(m.group("lam")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    rows = collect_rows(args.input_dir)
    if not rows:
        raise SystemExit(f"No sweep_summary.json files found under {args.input_dir}")
    write_csv(rows, args.output_csv)

    print(format_table(rows))
    all_msa_gt_mlp = all(float(r["msa_mean_compute_ratio"]) > float(r["mlp_mean_compute_ratio"]) for r in rows)
    print(f"MSA > MLP in all rows: {all_msa_gt_mlp}")
    ratios = [float(r["full_compute_ratio"]) for r in rows]
    print(f"full_compute_ratio range: {min(ratios):.4f} ~ {max(ratios):.4f}")
    print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
