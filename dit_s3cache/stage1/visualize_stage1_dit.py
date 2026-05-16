"""
Stage-1 baseline 可視化（DiT 版）：global cutting、zones、per-block k 與 expanded_mask。
"""

from __future__ import annotations
from typing import Any

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_stage1(output_dir: str) -> Any:
    """Public function load_stage1."""
    p = Path(output_dir)
    with open(p / "scheduler_config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(p / "scheduler_diagnostics.json", encoding="utf-8") as f:
        diag = json.load(f)
    return config, diag


def plot_global_cutting(config: Any, diag: Any, save_path: Path) -> Any:
    """G（processing order）平滑、Δ、change points、zones（步序 i，i=0 為 t=249）。"""
    T = int(config["T"])
    g_raw = np.array(diag["G_processing_order_i0_is_t249"], dtype=np.float64)
    g_sm = np.array(diag["G_smooth_processing_order"], dtype=np.float64)
    delta = np.array(diag["Delta_processing_order"], dtype=np.float64)
    cps = diag.get("change_points_step_index", [])
    zones = config["shared_zones"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    x = np.arange(T)
    axes[0].plot(x, g_raw, alpha=0.5, label="G (raw, proc. order)", lw=1)
    axes[0].plot(x, g_sm, lw=2, label="G smooth (moving avg)")
    for cp in cps:
        axes[0].axvline(cp, color="red", alpha=0.4, ls="--", lw=1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(zones), 1)))
    for zi, z in enumerate(zones):
        s0 = (T - 1) - int(z["t_start"])
        s1 = (T - 1) - int(z["t_end"])
        lo, hi = min(s0, s1), max(s0, s1)
        for ax in axes:
            ax.axvspan(lo, hi, alpha=0.12, color=colors[zi % 10])
    axes[0].set_ylabel("G")
    axes[0].set_title("Global cutting signal (FID-weighted I_cut) + smoothing + zones")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, delta, color="purple", lw=1)
    for cp in cps:
        axes[1].axvline(cp, color="red", alpha=0.4, ls="--", lw=1)
    axes[1].set_ylabel("|ΔG|")
    axes[1].set_xlabel("step index i (i=0 → t=249)")
    axes[1].set_title("Adjacent-step delta (on smoothed G)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {save_path}")


def plot_k_zone_heatmap(config: Any, save_path: Path) -> Any:
    """Public function plot_k_zone_heatmap."""
    blocks = config["blocks"]
    Z = len(config["shared_zones"])
    B = len(blocks)
    mat = np.zeros((B, Z), dtype=np.float64)
    names = []
    for bi, b in enumerate(blocks):
        mat[bi, :] = np.array(b["k_per_zone"], dtype=np.float64)
        names.append(str(b["name"])[:40])

    fig, ax = plt.subplots(figsize=(max(10, Z * 0.4), max(6, B * 0.2)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=1)
    ax.set_xticks(range(Z))
    zl = [f"z{z['id']}\nt{z['t_start']}→{z['t_end']}" for z in config["shared_zones"]]
    ax.set_xticklabels(zl, fontsize=7)
    ax.set_yticks(range(B))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("shared zones")
    ax.set_ylabel("blocks")
    ax.set_title("per-block k per shared zone")
    plt.colorbar(im, ax=ax, label="k")
    for bi in range(B):
        for zi in range(Z):
            ax.text(zi, bi, int(mat[bi, zi]), ha="center", va="center", fontsize=6, color="k")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {save_path}")


def plot_expanded_mask_heatmap(config: Any, save_path: Path) -> Any:
    """Public function plot_expanded_mask_heatmap."""
    T = int(config["T"])
    blocks = config["blocks"]
    B = len(blocks)
    mat = np.zeros((B, T), dtype=np.float64)
    for bi, b in enumerate(blocks):
        mat[bi, :] = np.array(b["expanded_mask"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(14, max(4, B * 0.18)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("step index i (0=t=249 → T-1=t=0)")
    ax.set_ylabel("block")
    ax.set_yticks(range(B))
    ax.set_yticklabels(
        [str(blocks[i]["name"])[:50] for i in range(B)], fontsize=5
    )
    ax.set_title("expanded_mask: F=1 (green) / R=0 (red)")
    plt.colorbar(im, ax=ax, ticks=[0, 1], label="F=1")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {save_path}")


def plot_candidate_selected_summary(config: Any, diag: Any, save_path: Path) -> Any:
    """每 zone：候選 k 數量 + 各 block 選中 k（小 multiples）。"""
    zones = config["shared_zones"]
    cands = diag.get("candidate_k_per_zone", [])
    Z = len(zones)
    blocks = config["blocks"]
    B = len(blocks)

    fig, axes = plt.subplots(Z, 1, figsize=(12, min(2.2 * Z, 24)), squeeze=False)
    for zi, z in enumerate(zones):
        ax = axes[zi, 0]
        ks = np.array([blocks[b]["k_per_zone"][zi] for b in range(B)], dtype=int)
        ax.scatter(range(B), ks, c="steelblue", s=20, label="selected k", zorder=3)
        if zi < len(cands):
            ax.text(
                0.02,
                0.95,
                f"|unique k cands| = {len(cands[zi])}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
            )
        ax.set_xlim(-0.5, B - 0.5)
        ax.set_ylabel("k")
        ax.set_title(f"zone {z['id']}  t=[{z['t_start']},{z['t_end']}]  L={z['length']}")
        ax.grid(alpha=0.3)
        if zi == Z - 1:
            ax.set_xlabel("block index")
    plt.suptitle("Selected k per block (per zone)", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {save_path}")


def main() -> Any:
    """Public function main."""
    parser = argparse.ArgumentParser(description="Visualize Stage-1 baseline (DiT)")
    parser.add_argument(
        "--stage1_output_dir",
        type=str,
        default="dit_s3cache/stage1/stage1_output",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dit_s3cache/stage1/stage1_figures",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config, diag = load_stage1(args.stage1_output_dir)

    plot_global_cutting(config, diag, out / "1_global_cutting_and_zones.png")
    plot_k_zone_heatmap(config, out / "2_k_zone_heatmap.png")
    plot_expanded_mask_heatmap(config, out / "3_expanded_mask_heatmap.png")
    plot_candidate_selected_summary(config, diag, out / "4_candidate_selected_k.png")
    print(f"完成，輸出目錄: {out}")


if __name__ == "__main__":
    main()
