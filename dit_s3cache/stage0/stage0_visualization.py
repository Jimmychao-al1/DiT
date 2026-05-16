"""
DiT Stage 0 Visualization

讀取 Stage 0 的 .npy 輸出（與 Diff-AE stage0e 相同檔名），繪製：
1. 每個 block 的三條曲線（L1 / cosine distance / Grassmann SVD distance）
2. 所有 block 的 FID weight bar chart
3. 全局 heatmap（B × T-1）

用法（在 DiT repo 根目錄）::

    python3 dit_s3cache/stage0/stage0_visualization.py
    python3 dit_s3cache/stage0/stage0_visualization.py \\
        --input-dir dit_s3cache/stage0/stage0_output \\
        --output-dir dit_s3cache/stage0/stage0_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# DiT-XL/2 defaults (T=250 → 249 intervals)
_DEFAULT_MODEL_LABEL = "DiT-XL/2, T=250"


def _build_default_t_curr(T_minus_1: int) -> np.ndarray:
    """Fallback: interval index j -> t_curr = (T-2) - j."""
    return (T_minus_1 - 1) - np.arange(T_minus_1, dtype=np.int32)


def load_stage0_outputs(output_dir: str) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """
    從 output_dir 讀取 Stage 0 結果（與 ``load_stage0_formal`` 相同檔案）。

    Returns:
        block_names, l1_norm, cos_norm, svd_norm, fid_w, t_curr, axis_def
    """
    p = Path(output_dir)
    required = [
        "block_names.npy",
        "l1_interval_norm.npy",
        "cosdist_interval_norm.npy",
        "svd_interval_norm.npy",
        "fid_w_qdiffae_clip.npy",
    ]
    for fname in required:
        if not (p / fname).is_file():
            raise FileNotFoundError(f"缺少 Stage 0 檔案: {p / fname}")

    block_names = np.load(p / "block_names.npy", allow_pickle=True)
    l1_norm = np.load(p / "l1_interval_norm.npy")
    cos_norm = np.load(p / "cosdist_interval_norm.npy")
    svd_norm = np.load(p / "svd_interval_norm.npy")
    fid_w = np.load(p / "fid_w_qdiffae_clip.npy")
    T_minus_1 = int(l1_norm.shape[1])

    t_curr_path = p / "t_curr_interval.npy"
    axis_def_path = p / "axis_interval_def.npy"
    if t_curr_path.is_file():
        t_curr = np.load(t_curr_path)
    else:
        t_curr = _build_default_t_curr(T_minus_1)
    if axis_def_path.is_file():
        axis_def_obj = np.load(axis_def_path, allow_pickle=True)
        axis_def = str(axis_def_obj.item() if hasattr(axis_def_obj, "item") else axis_def_obj)
    else:
        axis_def = "interval-wise: display label t_curr=(T-2)-j (fallback)"

    if t_curr.ndim != 1 or int(t_curr.shape[0]) != T_minus_1:
        t_curr = _build_default_t_curr(T_minus_1)
        axis_def = "interval-wise: display label t_curr=(T-2)-j (fallback due to invalid shape)"

    return block_names, l1_norm, cos_norm, svd_norm, fid_w, t_curr, axis_def


def _short_block_name(name: str) -> str:
    return str(name).replace("model.", "")


def plot_block_curves(
    block_name: str,
    l1: np.ndarray,
    cos: np.ndarray,
    svd: np.ndarray,
    fid_w: float,
    t_curr: np.ndarray,
    save_path: str,
) -> None:
    """單一 block：L1 / cosine distance / Grassmann distance 三曲線。"""
    T_minus_1 = len(l1)
    x = np.arange(T_minus_1)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, l1, label="L1 relative diff (norm)", color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax.plot(x, cos, label="Cosine distance (norm)", color="#ff7f0e", linewidth=1.2, alpha=0.9)
    ax.plot(
        x,
        svd,
        label="Grassmann distance (norm)",
        color="#2ca02c",
        linewidth=1.2,
        alpha=0.9,
    )

    ax.set_xlabel("Current timestep t_curr in transition x_{t+1} -> x_t", fontsize=10)
    ax.set_ylabel("Normalized value  [0, 1]", fontsize=10)
    ax.set_title(
        f"{block_name}    (FID weight = {fid_w:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, T_minus_1 - 1)
    y_hi = max(float(l1.max()), float(cos.max()), float(svd.max()))
    ax.set_ylim(-0.02, min(1.05, y_hi * 1.15 + 0.02))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    xticks = list(range(0, T_minus_1, 20))
    if (T_minus_1 - 1) not in xticks:
        xticks.append(T_minus_1 - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(t_curr[j])) for j in xticks])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_selected_blocks(
    block_names: np.ndarray,
    l1_norm: np.ndarray,
    cos_norm: np.ndarray,
    svd_norm: np.ndarray,
    fid_w: np.ndarray,
    t_curr: np.ndarray,
    indices: list[int],
    save_dir: str,
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        name = str(block_names[idx])
        slug = name.replace(".", "_")
        out = save_path / f"{slug}_curves.png"
        plot_block_curves(
            block_name=name,
            l1=l1_norm[idx],
            cos=cos_norm[idx],
            svd=svd_norm[idx],
            fid_w=float(fid_w[idx]),
            t_curr=t_curr,
            save_path=str(out),
        )
        print(f"  ✅ {name} -> {out}")


def plot_fid_weight_bar(
    block_names: np.ndarray,
    fid_w: np.ndarray,
    save_path: str,
    model_label: str = _DEFAULT_MODEL_LABEL,
) -> None:
    """所有 block 的 FID weight 橫條圖（按 weight 遞減）。"""
    B = len(block_names)
    order = np.argsort(fid_w)[::-1]

    names_sorted = [_short_block_name(block_names[i]) for i in order]
    w_sorted = fid_w[order]

    colors = []
    for w in w_sorted:
        if w > 0:
            intensity = 0.3 + 0.7 * w
            colors.append((0.12, 0.47 * intensity, 0.71 * intensity + 0.29 * (1 - intensity)))
        else:
            colors.append((0.75, 0.75, 0.75))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(B), w_sorted, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(B))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("FID weight  w_b  [0, 1]", fontsize=10)
    ax.set_title(
        f"Per-block FID Sensitivity Weight ({model_label})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", alpha=0.3)

    for i, (w, _bar) in enumerate(zip(w_sorted, ax.patches)):
        if w > 0:
            ax.text(w + 0.01, i, f"{w:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ FID weight bar -> {save_path}")


def plot_heatmap(
    data: np.ndarray,
    block_names: np.ndarray,
    title: str,
    save_path: str,
    t_curr: np.ndarray,
    cmap: str = "YlOrRd",
) -> None:
    """(B, T-1) heatmap；X 軸為 t_curr 顯示標籤。"""
    B, T = data.shape
    short_names = [_short_block_name(n) for n in block_names]

    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    ax.set_xlabel("Current timestep t_curr in transition x_{t+1} -> x_t", fontsize=10)
    ax.set_ylabel("Block", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yticks(range(B))
    ax.set_yticklabels(short_names, fontsize=7)

    xticks = list(range(0, T, 20))
    if (T - 1) not in xticks:
        xticks.append(T - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(t_curr[j]) for j in xticks], fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Normalized value [0, 1]", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Heatmap -> {save_path}")


def plot_combined_overview(
    block_idx: int,
    block_name: str,
    l1: np.ndarray,
    cos: np.ndarray,
    svd: np.ndarray,
    fid_w_all: np.ndarray,
    block_names: np.ndarray,
    t_curr: np.ndarray,
    save_path: str,
    model_label: str = _DEFAULT_MODEL_LABEL,
) -> None:
    """上半：三曲線；下半：FID weight bar（highlight 選定 block）。"""
    T_minus_1 = len(l1)
    x = np.arange(T_minus_1)
    B = len(block_names)
    order = np.argsort(-fid_w_all, kind="stable")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.3)

    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(x, l1, label="L1 relative diff", color="#1f77b4", linewidth=1.3)
    ax_top.plot(x, cos, label="Cosine distance", color="#ff7f0e", linewidth=1.3)
    ax_top.plot(x, svd, label="Grassmann distance", color="#2ca02c", linewidth=1.3)
    ax_top.set_xlabel("Current timestep t_curr in transition x_{t+1} -> x_t", fontsize=10)
    ax_top.set_ylabel("Normalized [0, 1]", fontsize=10)
    ax_top.set_title(
        f"Stage 0: {block_name}  (w_b = {fid_w_all[block_idx]:.4f}, {model_label})",
        fontsize=13,
        fontweight="bold",
    )
    ax_top.set_xlim(0, T_minus_1 - 1)
    xticks = list(range(0, T_minus_1, 20))
    if (T_minus_1 - 1) not in xticks:
        xticks.append(T_minus_1 - 1)
    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels([int(t_curr[j]) for j in xticks])
    y_max = max(float(l1.max()), float(cos.max()), float(svd.max()))
    ax_top.set_ylim(-0.02, min(1.05, y_max * 1.15 + 0.02))
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=0.3)

    ax_bot = fig.add_subplot(gs[1])
    names_sorted = [_short_block_name(block_names[i]) for i in order]
    w_sorted = fid_w_all[order]
    selected_pos = int(np.where(order == block_idx)[0][0])

    colors = []
    for i_sorted, orig_idx in enumerate(order):
        if orig_idx == block_idx:
            colors.append("#d62728")
        elif w_sorted[i_sorted] > 0:
            colors.append("#1f77b4")
        else:
            colors.append("#cccccc")

    ax_bot.barh(range(B), w_sorted, color=colors, edgecolor="white", linewidth=0.3)
    ax_bot.set_yticks(range(B))
    display_names = list(names_sorted)
    display_names[selected_pos] = f"> {display_names[selected_pos]}"
    ax_bot.set_yticklabels(display_names, fontsize=7)
    yticklabels = ax_bot.get_yticklabels()
    yticklabels[selected_pos].set_color("#d62728")
    yticklabels[selected_pos].set_fontweight("bold")
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("FID sensitivity weight w_b", fontsize=10)
    ax_bot.set_title("FID Sensitivity (red = selected block)", fontsize=11)
    ax_bot.set_xlim(0, 1.08)
    ax_bot.grid(axis="x", alpha=0.3)
    ax_bot.axhspan(selected_pos - 0.45, selected_pos + 0.45, color="#d62728", alpha=0.12, zorder=0)
    marker_x = max(float(w_sorted[selected_pos]) + 0.015, 0.015)
    ax_bot.scatter([marker_x], [selected_pos], color="#d62728", s=20, marker=">", zorder=3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Combined overview -> {save_path}")


def main(
    input_dir: str,
    output_dir: str,
    *,
    block_indices: list[int] | None = None,
    skip_overview: bool = False,
    model_label: str = _DEFAULT_MODEL_LABEL,
) -> None:
    """
    DiT Stage 0 可視化主流程。

    Args:
        input_dir: ``stage0_dit.py`` 輸出的 .npy 目錄
        output_dir: 圖片輸出目錄
        block_indices: 若指定，只畫這些 block；預設全部 B 個 block
        skip_overview: 若 True，略過 per-block combined overview（較快）
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DiT Stage 0 Visualization")
    print("=" * 60)

    print("\n[1] 載入 Stage 0 輸出...")
    block_names, l1_norm, cos_norm, svd_norm, fid_w, t_curr, axis_def = load_stage0_outputs(
        input_dir
    )
    B, T_minus_1 = l1_norm.shape
    print(f"    B={B}, T-1={T_minus_1}")
    print(f"    axis: {axis_def}")

    indices = list(block_indices) if block_indices is not None else list(range(B))
    print(f"\n[2] 繪圖 block 數: {len(indices)}")
    for idx in indices:
        tag = "HIGH" if fid_w[idx] > 0.1 else ("LOW" if fid_w[idx] == 0 else "MID")
        print(f"      [{idx:2d}] {block_names[idx]:12s}  w={fid_w[idx]:.4f}  ({tag})")

    print("\n[3] Per-block 三曲線...")
    curves_dir = out / "block_curves"
    plot_selected_blocks(
        block_names,
        l1_norm,
        cos_norm,
        svd_norm,
        fid_w,
        t_curr=t_curr,
        indices=indices,
        save_dir=str(curves_dir),
    )

    if not skip_overview:
        print("\n[4] Combined overview...")
        overview_dir = out / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            slug = str(block_names[idx]).replace(".", "_")
            plot_combined_overview(
                block_idx=idx,
                block_name=str(block_names[idx]),
                l1=l1_norm[idx],
                cos=cos_norm[idx],
                svd=svd_norm[idx],
                fid_w_all=fid_w,
                block_names=block_names,
                t_curr=t_curr,
                save_path=str(overview_dir / f"{slug}_overview.png"),
                model_label=model_label,
            )
    else:
        print("\n[4] 略過 combined overview (--skip-overview)")

    print("\n[5] FID weight bar chart...")
    plot_fid_weight_bar(
        block_names,
        fid_w,
        str(out / "fid_weight_bar.png"),
        model_label=model_label,
    )

    print("\n[6] 全局 heatmap...")
    heatmap_dir = out / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(
        l1_norm,
        block_names,
        "L1 relative diff (normalized)",
        str(heatmap_dir / "heatmap_l1.png"),
        t_curr=t_curr,
    )
    plot_heatmap(
        cos_norm,
        block_names,
        "Cosine distance (normalized)",
        str(heatmap_dir / "heatmap_cosdist.png"),
        t_curr=t_curr,
    )
    plot_heatmap(
        svd_norm,
        block_names,
        "Grassmann distance (normalized)",
        str(heatmap_dir / "heatmap_svd.png"),
        t_curr=t_curr,
    )

    print("\n" + "=" * 60)
    print(f"✅ 所有圖片已儲存至: {out}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    _repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="DiT S3-Cache Stage 0 visualization")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(_repo / "dit_s3cache/stage0/stage0_output"),
        help="Stage 0 .npy output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_repo / "dit_s3cache/stage0/stage0_figures"),
        help="Figure output directory",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=None,
        help="Optional block indices to plot (default: all blocks)",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip per-block combined overview figures (faster)",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default=_DEFAULT_MODEL_LABEL,
        help="Model label for plot titles",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    inp = Path(args.input_dir)
    if not inp.is_dir():
        raise FileNotFoundError(f"input-dir not found: {inp}")

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        block_indices=args.blocks,
        skip_overview=args.skip_overview,
        model_label=args.model_label,
    )
