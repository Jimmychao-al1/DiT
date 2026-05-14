"""Collect Stage 0 S3-Cache evidence for DiT-XL/2 256x256.

This script samples in latent space only.  It intentionally does not load the
VAE or decode images; the evidence target is each DiTBlock residual
``r_msa + r_mlp`` during the denoising trajectory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion import create_diffusion
from download import find_model
from models import DiT_models

from dit_s3cache.evidence.hooks import install_dit_block_hooks, restore_hooks
from dit_s3cache.evidence.utils import (
    EvidenceAccumulator,
    compute_similarity,
    compute_svd_drift,
    save_evidence_npz,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=list(DiT_models.keys()))
    parser.add_argument("--image-size", type=int, default=256, choices=[256])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--per-side-batch-size", type=int, default=4)
    parser.add_argument("--n-batches", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--k-svd", type=int, default=16)
    parser.add_argument("--svd-token-subsample", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--l1-denominator",
        type=str,
        default="current",
        choices=["current", "previous", "max", "symmetric"],
    )
    parser.add_argument("--output", type=Path, default=Path("dit_s3cache/outputs/evidence_dit_xl2_256.npz"))
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional smoke-test limit; by default run all diffusion steps.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run normally but also save final latent from each batch for manual decoding.",
    )
    parser.add_argument(
        "--sanity-output",
        type=Path,
        default=Path("dit_s3cache/outputs/sanity_latents.pt"),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    if args.model != "DiT-XL/2":
        raise ValueError("Stage 0 evidence collection is currently scoped to DiT-XL/2.")
    if args.image_size != 256:
        raise ValueError("Stage 0 evidence collection is currently scoped to 256x256 only.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    model.load_state_dict(find_model(ckpt_path))
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    n_steps = diffusion.num_timesteps if args.max_steps is None else min(args.max_steps, diffusion.num_timesteps)
    timestep_map = list(range(diffusion.num_timesteps))[::-1][:n_steps]
    n_blocks = len(model.blocks)

    storage: dict[int, torch.Tensor] = {}
    hooks = install_dit_block_hooks(model, storage)
    accumulator = EvidenceAccumulator(n_blocks=n_blocks, n_steps=n_steps, k_svd=args.k_svd)
    seeds: list[int] = []
    sanity_latents: list[torch.Tensor] = []
    n_batches_to_run = 1 if args.sanity_check else args.n_batches

    try:
        for batch_idx in range(n_batches_to_run):
            seed = args.base_seed + batch_idx
            seeds.append(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            z, y = make_cfg_inputs(
                batch_size=args.per_side_batch_size,
                latent_size=latent_size,
                num_classes=args.num_classes,
                device=device,
            )
            model_kwargs = {"y": y, "cfg_scale": args.cfg_scale}
            prev_residuals: list[torch.Tensor | None] = [None] * n_blocks
            v_prevs: list[torch.Tensor | None] = [None] * n_blocks
            final_latent: torch.Tensor | None = None

            sample_iter = diffusion.p_sample_loop_progressive(
                model.forward_with_cfg,
                z.shape,
                noise=z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=device,
                progress=args.progress,
            )

            for step_idx, out in enumerate(sample_iter):
                if step_idx >= n_steps:
                    break
                final_latent = out["sample"].detach()
                ensure_complete_storage(storage, n_blocks, step_idx)

                # SVD/subsampling consume torch RNG. Restore it before the next
                # diffusion step so evidence collection does not change samples.
                with preserve_torch_rng():
                    for block_idx in range(n_blocks):
                        residual = storage[block_idx]
                        prev_residual = prev_residuals[block_idx]
                        if prev_residual is not None:
                            sim = compute_similarity(
                                residual,
                                prev_residual,
                                denominator=args.l1_denominator,
                            )
                            accumulator.add_similarity(block_idx, step_idx, sim)

                        svd = compute_svd_drift(
                            residual,
                            v_prevs[block_idx],
                            k=args.k_svd,
                            token_subsample=args.svd_token_subsample or None,
                        )
                        accumulator.add_svd(block_idx, step_idx, svd)

                        prev_residuals[block_idx] = residual.detach().clone()
                        v_prevs[block_idx] = svd["v_current"]

                storage.clear()

            if args.sanity_check and final_latent is not None:
                sanity_latents.append(final_latent.cpu())

            del prev_residuals, v_prevs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Batch {batch_idx + 1}/{n_batches_to_run} done")

    finally:
        restore_hooks(hooks)

    arrays = accumulator.finalize()
    arrays["timestep_map"] = np.asarray(timestep_map, dtype=np.int64)
    metadata = build_metadata(
        args,
        n_blocks=n_blocks,
        n_steps=n_steps,
        n_batches_run=n_batches_to_run,
        seeds=seeds,
    )
    save_evidence_npz(args.output, arrays, metadata)
    print(f"Saved evidence to {args.output}")

    if args.sanity_check and sanity_latents:
        args.sanity_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latents": torch.stack(sanity_latents),
                "metadata": metadata,
            },
            args.sanity_output,
        )
        print(f"Saved sanity latents to {args.sanity_output}")


def make_cfg_inputs(
    batch_size: int,
    latent_size: int,
    num_classes: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    z = torch.cat([z, z], dim=0)
    y_null = torch.tensor([num_classes] * batch_size, device=device)
    y = torch.cat([y, y_null], dim=0)
    return z, y


def ensure_complete_storage(storage: dict[int, torch.Tensor], n_blocks: int, step_idx: int) -> None:
    missing = [idx for idx in range(n_blocks) if idx not in storage]
    if missing:
        raise RuntimeError(f"Missing residuals at step {step_idx}: blocks={missing}")


@contextmanager
def preserve_torch_rng():
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def build_metadata(
    args: argparse.Namespace,
    n_blocks: int,
    n_steps: int,
    n_batches_run: int,
    seeds: list[int],
) -> dict[str, Any]:
    return {
        "stage": "S3-Cache Stage 0 Evidence Collection",
        "format": "dit_s3cache_v1",
        "model": args.model,
        "image_size": args.image_size,
        "latent_size": args.image_size // 8,
        "token_count": 256,
        "hidden_size": 1152,
        "n_blocks": n_blocks,
        "n_steps_recorded": n_steps,
        "num_sampling_steps": args.num_sampling_steps,
        "timestep_indexing": "array step_idx follows sampling order; timestep_map[step_idx] gives diffusion timestep",
        "per_side_batch_size": args.per_side_batch_size,
        "effective_model_batch_size": 2 * args.per_side_batch_size,
        "n_batches_requested": args.n_batches,
        "n_batches_run": n_batches_run,
        "cfg_scale": args.cfg_scale,
        "cfg_strategy": "cond/uncond are batched and averaged together for evidence",
        "k_svd": args.k_svd,
        "svd_token_subsample": args.svd_token_subsample or None,
        "l1_denominator": args.l1_denominator,
        "evidence_targets": {
            "block_residual": "r_msa + r_mlp after adaLN-Zero gates",
            "similarity": ["l1_diff", "cos_sim"],
            "svd": ["grassmann_dist", "subspace_overlap", "sv_spectrum"],
        },
        "cross_model_semantic_mapping": {
            "Diff-AE/LDM block residual evidence": "DiT DiTBlock residual r_msa + r_mlp",
            "spatial feature flattening": "DiT token-by-hidden flattening",
            "Diff-AE/LDM l1": "symmetric mean absolute relative difference; run DiT with --l1-denominator symmetric to match this normalization",
            "Diff-AE/LDM l1_rate": "asymmetric ||current-previous||_1 / ||previous||_1; run DiT with --l1-denominator previous to match this normalization",
            "stage1_note": "DiT uses its own key names and metadata because block topology and CFG differ.",
        },
        "seeds": seeds,
        "script_args": json.loads(json.dumps(_jsonable_args(args))),
    }


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value)
    return result


if __name__ == "__main__":
    main(build_argparser().parse_args())
