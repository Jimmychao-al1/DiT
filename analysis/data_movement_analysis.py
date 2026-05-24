#!/usr/bin/env python3
"""DiT-XL/2 data-movement and CIM array analysis.

Run with:
  /home/jimmy/anaconda3/envs/ldm/bin/python data_movement_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_KEY = "dit_xl2"
REPO_ROOT = Path("/home/jimmy/DiT")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "output" / "data_movement"

MODEL_NAME = "DiT-XL/2"
CKPT_PATH = REPO_ROOT / "pretrained_models/DiT-XL-2-256x256.pt"
SCHEDULER_PATH = (
    REPO_ROOT
    / "dit_s3cache/stage2/stage2_output_ddim150/"
    / "src_baseline_p2_K20_sw3_lam1.0_kmax3/02_refined_blockwise/"
    / "stage2_refined_scheduler_config.json"
)

T_EXPECTED = 150
BYTES_PER_ELEMENT = 4
NUM_BLOCKS = 28
NUM_TOKENS = 256
HIDDEN_DIM = 1152
IMAGE_SIZE = 256
BATCH_SIZE = 1


@dataclass
class LayerRecord:
    model: str
    block_id: int
    block_name: str
    layer_name: str
    layer_type: str
    weight_shape: str
    weight_bytes: int
    act_bytes: int
    dm_per_exec: int
    cim_rows: int
    cim_cols: int
    cim_max_dim: int
    cim_array_size: int
    is_quantized: bool
    seq_len: int


@dataclass
class BlockRecord:
    model: str
    block_id: int
    block_name: str
    canonical_name: str
    spatial_h: int
    spatial_w: int
    input_channels: int
    output_channels: int
    weight_bytes_per_exec: int
    act_bytes_per_exec: int
    dm_per_exec: int
    exec_count_baseline: int
    exec_count_cached: int
    dm_baseline: int
    dm_cached: int
    cim_block_max_dim: int
    cim_block_max_array_size: int


def format_bytes(n: int | float) -> str:
    n = int(n)
    if n >= 1 << 30:
        return f"{n / (1 << 30):.4f} GB ({n:,} B)"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.4f} MB ({n:,} B)"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.4f} KB ({n:,} B)"
    return f"{n:,} B"


def byte_summary_fields(prefix: str, n: int | float) -> dict[str, Any]:
    n = float(n)
    return {
        f"{prefix}_readable": format_bytes(n),
        f"{prefix}_KB": n / (1 << 10),
        f"{prefix}_MB": n / (1 << 20),
        f"{prefix}_GB": n / (1 << 30),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    rows = [[str(x) for x in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def runtime_name_from_id(block_id: int) -> str:
    return f"block_{block_id}"


def load_scheduler(path: Path) -> tuple[int, list[dict[str, Any]]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    t = int(cfg["T"])
    if t != T_EXPECTED:
        raise ValueError(f"Scheduler T={t}, expected {T_EXPECTED}")
    blocks = sorted(cfg["blocks"], key=lambda b: int(b["canonical_runtime_block_id"]))
    seen = [int(b["canonical_runtime_block_id"]) for b in blocks]
    if seen != list(range(NUM_BLOCKS)):
        raise ValueError(f"canonical_runtime_block_id must be 0..{NUM_BLOCKS - 1}, got {seen}")
    for b in blocks:
        runtime = str(b["runtime_name"])
        if runtime != runtime_name_from_id(int(b["canonical_runtime_block_id"])):
            raise ValueError(f"runtime name/id mismatch in scheduler block: {b}")
        if len(b["expanded_mask"]) != t:
            raise ValueError(f"{runtime}: mask length mismatch")
    return t, blocks


def load_model() -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    import torch
    from models import DiT_models

    latent_size = IMAGE_SIZE // 8
    model = DiT_models[MODEL_NAME](input_size=latent_size, num_classes=1000)
    state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "ema" in state:
        state = state["ema"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def linear_record(*, block_id: int, block_name: str, layer_name: str, module: Any, seq_len: int) -> LayerRecord:
    out_f, in_f = [int(x) for x in module.weight.shape]
    weight_bytes = out_f * in_f * BYTES_PER_ELEMENT
    act_bytes = BATCH_SIZE * seq_len * in_f * BYTES_PER_ELEMENT
    return LayerRecord(
        model=MODEL_KEY,
        block_id=block_id,
        block_name=block_name,
        layer_name=layer_name,
        layer_type="linear",
        weight_shape=str((out_f, in_f)),
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        dm_per_exec=weight_bytes + act_bytes,
        cim_rows=in_f,
        cim_cols=out_f,
        cim_max_dim=max(in_f, out_f),
        cim_array_size=in_f * out_f,
        is_quantized=False,
        seq_len=seq_len,
    )


def enumerate_block(block: Any, *, block_id: int, block_name: str) -> list[LayerRecord]:
    import torch.nn as nn

    records: list[LayerRecord] = []
    expected_names = {"attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2", "adaLN_modulation.1"}
    for name, module in block.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in expected_names:
            raise ValueError(f"Unexpected Linear in {block_name}: {name}")
        seq_len = 1 if name == "adaLN_modulation.1" else NUM_TOKENS
        records.append(
            linear_record(
                block_id=block_id,
                block_name=block_name,
                layer_name=f"{block_name}.{name}",
                module=module,
                seq_len=seq_len,
            )
        )
    got = {r.layer_name.split(f"{block_name}.", 1)[1] for r in records}
    if got != expected_names:
        raise ValueError(f"{block_name}: expected linear names {expected_names}, got {got}")
    return records


def analyze() -> tuple[list[BlockRecord], list[LayerRecord], dict[str, Any]]:
    t, sched_blocks = load_scheduler(SCHEDULER_PATH)
    model = load_model()
    if len(model.blocks) != NUM_BLOCKS:
        raise ValueError(f"Expected {NUM_BLOCKS} DiT blocks, got {len(model.blocks)}")

    layer_records: list[LayerRecord] = []
    block_records: list[BlockRecord] = []
    for b in sched_blocks:
        block_id = int(b["canonical_runtime_block_id"])
        block_name = str(b["runtime_name"])
        block = model.blocks[block_id]
        recs = enumerate_block(block, block_id=block_id, block_name=block_name)
        layer_records.extend(recs)
        weight_sum = sum(r.weight_bytes for r in recs)
        act_sum = sum(r.act_bytes for r in recs)
        exec_cached = sum(bool(x) for x in b["expanded_mask"])
        dm_exec = weight_sum + act_sum
        block_records.append(
            BlockRecord(
                model=MODEL_KEY,
                block_id=block_id,
                block_name=block_name,
                canonical_name=str(b["name"]),
                spatial_h=1,
                spatial_w=NUM_TOKENS,
                input_channels=HIDDEN_DIM,
                output_channels=HIDDEN_DIM,
                weight_bytes_per_exec=weight_sum,
                act_bytes_per_exec=act_sum,
                dm_per_exec=dm_exec,
                exec_count_baseline=t,
                exec_count_cached=exec_cached,
                dm_baseline=dm_exec * t,
                dm_cached=dm_exec * exec_cached,
                cim_block_max_dim=max(r.cim_max_dim for r in recs),
                cim_block_max_array_size=max(r.cim_array_size for r in recs),
            )
        )

    validate_records(t, block_records, layer_records)
    return block_records, layer_records, make_summary(t, block_records, layer_records)


def validate_records(t: int, blocks: list[BlockRecord], layers: list[LayerRecord]) -> None:
    if len(blocks) != NUM_BLOCKS:
        raise ValueError(f"Expected {NUM_BLOCKS} blocks, got {len(blocks)}")
    expected = [runtime_name_from_id(i) for i in range(NUM_BLOCKS)]
    if [b.block_name for b in blocks] != expected:
        raise ValueError("Runtime order mismatch")
    if len(layers) != NUM_BLOCKS * 5:
        raise ValueError(f"Expected {NUM_BLOCKS * 5} linear layers, got {len(layers)}")
    for r in layers:
        if not r.layer_name.startswith(r.block_name + "."):
            raise ValueError(f"Layer name not under block: {r.block_name} vs {r.layer_name}")
        if r.dm_per_exec != r.weight_bytes + r.act_bytes:
            raise ValueError(f"Invalid total bytes for {r.layer_name}")
        if r.layer_name.endswith("adaLN_modulation.1") and r.seq_len != 1:
            raise ValueError(f"adaLN seq_len mismatch for {r.layer_name}")
        if not r.layer_name.endswith("adaLN_modulation.1") and r.seq_len != NUM_TOKENS:
            raise ValueError(f"token seq_len mismatch for {r.layer_name}")
    for b in blocks:
        child = [r for r in layers if r.block_id == b.block_id]
        if b.weight_bytes_per_exec != sum(r.weight_bytes for r in child):
            raise ValueError(f"Weight sum mismatch for {b.block_name}")
        if b.act_bytes_per_exec != sum(r.act_bytes for r in child):
            raise ValueError(f"Activation sum mismatch for {b.block_name}")
        if b.dm_baseline != b.dm_per_exec * t:
            raise ValueError(f"Baseline mismatch for {b.block_name}")
        if b.dm_cached != b.dm_per_exec * b.exec_count_cached:
            raise ValueError(f"Cached mismatch for {b.block_name}")


def make_summary(t: int, blocks: list[BlockRecord], layers: list[LayerRecord]) -> dict[str, Any]:
    baseline = sum(b.dm_baseline for b in blocks)
    cached = sum(b.dm_cached for b in blocks)
    baseline_per_step = baseline / t
    cached_per_step = cached / t
    summary = {
        "model": MODEL_KEY,
        "T": t,
        "num_blocks": len(blocks),
        "num_layers": len(layers),
        "num_quantized_layers": 0,
        "bytes_per_element": BYTES_PER_ELEMENT,
        "baseline_bytes": baseline,
        "cached_bytes": cached,
        "reduction_ratio": (baseline - cached) / baseline,
        "baseline_bytes_per_step": baseline_per_step,
        "cached_bytes_per_step": cached_per_step,
        "global_cim_max_dim": max(r.cim_max_dim for r in layers),
        "global_cim_max_array_size": max(r.cim_array_size for r in layers),
    }
    summary.update(byte_summary_fields("baseline", baseline))
    summary.update(byte_summary_fields("cached", cached))
    summary.update(byte_summary_fields("baseline_per_step", baseline_per_step))
    summary.update(byte_summary_fields("cached_per_step", cached_per_step))
    return summary


def save_outputs(output_dir: Path, blocks: list[BlockRecord], layers: list[LayerRecord], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "exp1_block_detail.csv", [asdict(b) for b in blocks])
    write_csv(output_dir / "exp1_layer_detail.csv", [asdict(r) for r in layers])
    write_csv(output_dir / "exp2_cim_layer_detail.csv", [asdict(r) for r in layers])
    write_csv(
        output_dir / "exp2_cim_block_summary.csv",
        [
            {
                "model": b.model,
                "block_id": b.block_id,
                "block_name": b.block_name,
                "num_layers": sum(1 for r in layers if r.block_id == b.block_id),
                "block_max_dim": b.cim_block_max_dim,
                "block_max_array_size": b.cim_block_max_array_size,
                "spatial_h": b.spatial_h,
                "spatial_w": b.spatial_w,
                "input_channels": b.input_channels,
                "output_channels": b.output_channels,
            }
            for b in blocks
        ],
    )
    write_csv(output_dir / "exp1_summary.csv", [summary])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_report(blocks: list[BlockRecord], layers: list[LayerRecord], summary: dict[str, Any]) -> None:
    print(f"\nModel: {MODEL_KEY}")
    print(f"Baseline: {format_bytes(summary['baseline_bytes'])}")
    print(f"Cached:   {format_bytes(summary['cached_bytes'])}")
    print(f"Reduction: {summary['reduction_ratio']:.2%}")
    print(f"CIM max dim: {summary['global_cim_max_dim']}")
    print(f"CIM max array size: {summary['global_cim_max_array_size']:,}")
    print_table(
        "Block-Level Data Movement",
        ["id", "block", "tokens", "DM/exec", "exec", "cached DM", "CIM"],
        [
            [
                b.block_id,
                b.block_name,
                b.spatial_w,
                format_bytes(b.dm_per_exec),
                b.exec_count_cached,
                format_bytes(b.dm_cached),
                b.cim_block_max_dim,
            ]
            for b in blocks
        ],
    )
    print_table(
        "Layer-Level Data Movement",
        ["block", "layer", "seq", "weight", "act", "total", "CIM(r,c)"],
        [
            [
                r.block_name,
                r.layer_name,
                r.seq_len,
                format_bytes(r.weight_bytes),
                format_bytes(r.act_bytes),
                format_bytes(r.dm_per_exec),
                f"({r.cim_rows},{r.cim_cols})",
            ]
            for r in layers
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    blocks, layers, summary = analyze()
    save_outputs(Path(args.output_dir), blocks, layers, summary)
    print_report(blocks, layers, summary)
    print(f"\n[Done] Results written to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
