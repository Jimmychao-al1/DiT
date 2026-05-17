"""Collect Stage 0 S3-Cache evidence for DiT sub-layers.

This is the sub-layer counterpart of ``collect_evidence.py``.  It records the
gated MSA and MLP residual branches separately:

    block_00_msa, block_00_mlp, ..., block_27_mlp
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

from dit_s3cache.evidence.hooks_sublayer import (
    BRANCHES,
    install_dit_sublayer_hooks,
    restore_sublayer_hooks,
    sublayer_name,
)
from dit_s3cache.evidence.utils import (
    compute_similarity,
    compute_svd_drift,
    save_evidence_npz,
)


BRANCH_TO_INDEX = {"msa": 0, "mlp": 1}


class SubLayerEvidenceAccumulator:
    """NaN-safe accumulator for per-block, per-branch, per-step evidence."""

    def __init__(self, n_blocks: int, n_steps: int, k_svd: int) -> None:
        self.n_blocks = int(n_blocks)
        self.n_steps = int(n_steps)
        self.k_svd = int(k_svd)
        shape = (self.n_blocks, len(BRANCHES), self.n_steps)
        self.l1_sum = np.zeros(shape, dtype=np.float64)
        self.cos_sum = np.zeros(shape, dtype=np.float64)
        self.grassmann_sum = np.zeros(shape, dtype=np.float64)
        self.overlap_sum = np.zeros(shape, dtype=np.float64)
        self.sv_sum = np.zeros((*shape, self.k_svd), dtype=np.float64)
        self.pair_count = np.zeros(shape, dtype=np.int64)
        self.sv_count = np.zeros(shape, dtype=np.int64)

    def add_similarity(
        self,
        block_idx: int,
        branch: str,
        step_idx: int,
        metrics: dict[str, float],
    ) -> None:
        bidx = BRANCH_TO_INDEX[branch]
        if np.isnan(metrics["l1_diff"]) or np.isnan(metrics["cos_sim"]):
            return
        self.l1_sum[block_idx, bidx, step_idx] += metrics["l1_diff"]
        self.cos_sum[block_idx, bidx, step_idx] += metrics["cos_sim"]
        self.pair_count[block_idx, bidx, step_idx] += 1

    def add_svd(
        self,
        block_idx: int,
        branch: str,
        step_idx: int,
        metrics: dict[str, Any],
    ) -> None:
        bidx = BRANCH_TO_INDEX[branch]
        spectrum = np.asarray(metrics["sv_spectrum"], dtype=np.float64)
        self.sv_sum[block_idx, bidx, step_idx] += np.nan_to_num(spectrum, nan=0.0)
        self.sv_count[block_idx, bidx, step_idx] += 1

        if np.isnan(metrics["grassmann_dist"]) or np.isnan(metrics["subspace_overlap"]):
            return
        self.grassmann_sum[block_idx, bidx, step_idx] += metrics["grassmann_dist"]
        self.overlap_sum[block_idx, bidx, step_idx] += metrics["subspace_overlap"]

    def finalize(self) -> dict[str, np.ndarray]:
        pair_count = self.pair_count
        sv_count = self.sv_count
        l1 = _divide_or_nan(self.l1_sum, pair_count)
        cos = _divide_or_nan(self.cos_sum, pair_count)
        grassmann = _divide_or_nan(self.grassmann_sum, pair_count)
        overlap = _divide_or_nan(self.overlap_sum, pair_count)
        spectrum = _divide_or_nan(self.sv_sum, sv_count[..., None])

        names = [
            sublayer_name(block_idx, branch)
            for block_idx in range(self.n_blocks)
            for branch in BRANCHES
        ]

        arrays: dict[str, np.ndarray] = {
            "sub_layer_names": np.asarray(names, dtype=object),
            "branch_names": np.asarray(BRANCHES, dtype=object),
            "l1_diff_sublayer": _flatten_branches(l1),
            "cos_sim_sublayer": _flatten_branches(cos),
            "grassmann_dist_sublayer": _flatten_branches(grassmann),
            "subspace_overlap_sublayer": _flatten_branches(overlap),
            "sv_spectrum_sublayer": _flatten_branches(spectrum),
            "pair_count_sublayer": _flatten_branches(pair_count),
            "sv_count_sublayer": _flatten_branches(sv_count),
        }

        for block_idx in range(self.n_blocks):
            for branch in BRANCHES:
                bidx = BRANCH_TO_INDEX[branch]
                prefix = sublayer_name(block_idx, branch)
                arrays[f"{prefix}_l1_diff"] = l1[block_idx, bidx]
                arrays[f"{prefix}_cos_sim"] = cos[block_idx, bidx]
                arrays[f"{prefix}_grassmann_dist"] = grassmann[block_idx, bidx]
                arrays[f"{prefix}_subspace_overlap"] = overlap[block_idx, bidx]
                arrays[f"{prefix}_sv_spectrum"] = spectrum[block_idx, bidx]
                arrays[f"{prefix}_pair_count"] = pair_count[block_idx, bidx].copy()
                arrays[f"{prefix}_sv_count"] = sv_count[block_idx, bidx].copy()

        return arrays


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=list(DiT_models.keys()))
    parser.add_argument("--image-size", type=int, default=256, choices=[256])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--per-side-batch-size", type=int, default=4)
    parser.add_argument("--n-batches", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--k-svd", type=int, default=16)
    parser.add_argument("--svd-token-subsample", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--l1-denominator",
        type=str,
        default="current",
        choices=["current", "previous", "max", "symmetric"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dit_s3cache/outputs/evidence_dit_xl2_256_sublayer.npz"),
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument(
        "--sanity-output",
        type=Path,
        default=Path("dit_s3cache/outputs/sanity_latents_sublayer.pt"),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    if args.model != "DiT-XL/2":
        raise ValueError("Sub-layer evidence collection is currently scoped to DiT-XL/2.")
    if args.image_size != 256:
        raise ValueError("Sub-layer evidence collection is currently scoped to 256x256 only.")

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
    timestep_map = list(reversed(diffusion.timestep_map))[:n_steps]
    n_blocks = len(model.blocks)

    storage: dict[int, dict[str, torch.Tensor]] = {}
    hooks = install_dit_sublayer_hooks(model, storage)
    accumulator = SubLayerEvidenceAccumulator(n_blocks=n_blocks, n_steps=n_steps, k_svd=args.k_svd)
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
            prev_residuals: list[dict[str, torch.Tensor | None]] = [
                {branch: None for branch in BRANCHES} for _ in range(n_blocks)
            ]
            v_prevs: list[dict[str, torch.Tensor | None]] = [
                {branch: None for branch in BRANCHES} for _ in range(n_blocks)
            ]
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
                ensure_complete_sublayer_storage(storage, n_blocks, step_idx)

                with preserve_torch_rng():
                    for block_idx in range(n_blocks):
                        for branch in BRANCHES:
                            residual = storage[block_idx][branch]
                            prev_residual = prev_residuals[block_idx][branch]
                            if prev_residual is not None:
                                sim = compute_similarity(
                                    residual,
                                    prev_residual,
                                    denominator=args.l1_denominator,
                                )
                                accumulator.add_similarity(block_idx, branch, step_idx, sim)

                            svd = compute_svd_drift(
                                residual,
                                v_prevs[block_idx][branch],
                                k=args.k_svd,
                                token_subsample=args.svd_token_subsample or None,
                            )
                            accumulator.add_svd(block_idx, branch, step_idx, svd)

                            prev_residuals[block_idx][branch] = residual.detach().clone()
                            v_prevs[block_idx][branch] = svd["v_current"]

                storage.clear()

            if args.sanity_check and final_latent is not None:
                sanity_latents.append(final_latent.cpu())

            del prev_residuals, v_prevs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Batch {batch_idx + 1}/{n_batches_to_run} done")

    finally:
        restore_sublayer_hooks(hooks)

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
    print(f"Saved sub-layer evidence to {args.output}")

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


def ensure_complete_sublayer_storage(
    storage: dict[int, dict[str, torch.Tensor]],
    n_blocks: int,
    step_idx: int,
) -> None:
    missing: list[str] = []
    for block_idx in range(n_blocks):
        branches = storage.get(block_idx, {})
        for branch in BRANCHES:
            if branch not in branches:
                missing.append(sublayer_name(block_idx, branch))
    if missing:
        raise RuntimeError(f"Missing sub-layer residuals at step {step_idx}: {missing}")


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
        "stage": "S3-Cache Stage 0 Sub-layer Evidence Collection",
        "format": "dit_s3cache_sublayer_v1",
        "model": args.model,
        "image_size": args.image_size,
        "latent_size": args.image_size // 8,
        "token_count": 256,
        "hidden_size": 1152,
        "n_blocks": n_blocks,
        "n_sub_layers": n_blocks * len(BRANCHES),
        "branches": list(BRANCHES),
        "sub_layer_order": "block-major: block_00_msa, block_00_mlp, ...",
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
            "msa": "r_msa after adaLN-Zero gate",
            "mlp": "r_mlp after adaLN-Zero gate, with x including the current MSA branch",
            "similarity": ["l1_diff", "cos_sim"],
            "svd": ["grassmann_dist", "subspace_overlap", "sv_spectrum"],
        },
        "seeds": seeds,
        "script_args": json.loads(json.dumps(_jsonable_args(args))),
    }


def _flatten_branches(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]).copy()


def _divide_or_nan(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    return np.divide(values, counts, out=out, where=counts > 0)


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value)
    return result


if __name__ == "__main__":
    main(build_argparser().parse_args())
