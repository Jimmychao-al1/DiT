"""Q-DiT S3-Cache c_FID cache sensitivity experiments.

This reuses the DiT c_FID pipeline (PNG -> ADM NPZ -> evaluator), but swaps
model loading to Q-DiT W8A8.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import torch
from diffusers.models import AutoencoderKL

import dit_s3cache.fid.fid_cache_sensitivity as base


def build_argparser() -> argparse.ArgumentParser:
    parser = base.build_argparser()
    parser.set_defaults(
        num_sampling_steps=50,
        sampler="ddim",
        results_json=Path("dit_s3cache/fid/fid_sensitivity_results_qdit.json"),
        gen_image_dir=Path("dit_s3cache/fid/gen_image_qdit"),
        sample_npz=Path("dit_s3cache/fid/generated_samples_qdit.npz"),
    )
    parser.add_argument(
        "--qdit-ckpt",
        type=Path,
        default=Path("/home/jimmy/Q-DiT/results/003-qdit_w8a8/qdit_w8a8_state_dict.pt"),
        help="Quantized Q-DiT state_dict path.",
    )
    parser.add_argument(
        "--qdit-root",
        type=Path,
        default=Path("/home/jimmy/Q-DiT"),
        help="Q-DiT repository root used for imports.",
    )
    parser.add_argument("--wbits", type=int, default=8)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--weight-group-size", type=int, default=128)
    parser.add_argument("--act-group-size", type=int, default=128)
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=None,
        help="Optional local VAE directory. If unset, load from huggingface.",
    )
    return parser


def _prepend_qdit_root(qdit_root: Path) -> None:
    root = str(qdit_root.resolve())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def _import_qdit_modules(qdit_root: Path):
    """Import Q-DiT modules even if DiT already imported `models` module."""
    _prepend_qdit_root(qdit_root)
    saved_models = sys.modules.pop("models", None)
    saved_models_models = sys.modules.pop("models.models", None)
    try:
        diffusion_mod = importlib.import_module("diffusion")
        models_mod = importlib.import_module("models.models")
        modelutils_mod = importlib.import_module("qdit.modelutils")
        return diffusion_mod, models_mod, modelutils_mod
    finally:
        sys.modules.pop("models", None)
        sys.modules.pop("models.models", None)
        if saved_models is not None:
            sys.modules["models"] = saved_models
        if saved_models_models is not None:
            sys.modules["models.models"] = saved_models_models


def _build_qdit_args(args: argparse.Namespace, n_blocks: int) -> SimpleNamespace:
    quant_args = SimpleNamespace(
        wbits=args.wbits,
        abits=args.abits,
        exponential=False,
        quantize_bmm_input=False,
        a_sym=False,
        w_sym=False,
        static=False,
        weight_group_size=args.weight_group_size,
        weight_channel_group=1,
        act_group_size=args.act_group_size,
        tiling=0,
        quant_method="max",
        a_clip_ratio=1.0,
        w_clip_ratio=1.0,
        kv_clip_ratio=1.0,
        quant_type="int",
    )
    quant_args.weight_group_size = [quant_args.weight_group_size] * n_blocks
    quant_args.act_group_size = [quant_args.act_group_size] * n_blocks
    return quant_args


def load_everything_qdit(
    args: argparse.Namespace,
    device: str,
) -> tuple[torch.nn.Module, Any, AutoencoderKL]:
    diffusion_mod, models_mod, modelutils_mod = _import_qdit_modules(args.qdit_root)
    create_diffusion = diffusion_mod.create_diffusion
    DiT_models = models_mod.DiT_models
    add_act_quant_wrapper = modelutils_mod.add_act_quant_wrapper

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    qargs = _build_qdit_args(args, n_blocks=len(model.blocks))
    model = add_act_quant_wrapper(
        model,
        device=device,
        args=qargs,
        scales=defaultdict(lambda: None),
    )

    if not args.qdit_ckpt.is_file():
        raise FileNotFoundError(f"Q-DiT checkpoint not found: {args.qdit_ckpt}")
    state_dict = torch.load(args.qdit_ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    if args.vae_path is not None:
        vae = AutoencoderKL.from_pretrained(str(args.vae_path)).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    return model, diffusion, vae


def main(args: argparse.Namespace) -> None:
    # Reuse original main flow while swapping model loader.
    base.load_everything = load_everything_qdit  # type: ignore[assignment]
    base.main(args)


if __name__ == "__main__":
    main(build_argparser().parse_args())
