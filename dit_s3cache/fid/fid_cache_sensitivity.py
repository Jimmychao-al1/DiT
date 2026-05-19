"""DiT S3-Cache c_FID cache sensitivity experiments.

Each cached evaluation enables runtime residual caching for one DiTBlock while
all other blocks recompute normally.  Images are generated under
``dit_s3cache/fid/gen_image`` and converted to an ADM-compatible NPZ batch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
import subprocess
import sys
sys.path.append('.')
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm.auto import tqdm

from diffusion import create_diffusion
from download import find_model
from models import DiT_models

from dit_s3cache.fid.cache_runner import (
    cache_stats,
    create_dit_cache_config,
    install_cache_wrappers,
    make_cached_forward_with_cfg,
    reset_cache_state,
    restore_cache_wrappers,
)

# 與 guided-diffusion ADM evaluator 相同 CLI：``ref_batch sample_batch``
DEFAULT_ADM_EVALUATOR = Path(__file__).resolve().parent / "evaluator.py"
DEFAULT_REF_BATCH = Path(__file__).resolve().parent / "VIRTUAL_imagenet256_labeled.npz"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-fid-samples", type=int, default=1000)
    parser.add_argument("--per-side-batch-size", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta; only used with --sampler ddim.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--part",
        type=str,
        choices=["A", "B", "A2", "B2"],
        default=None,
        help="A/B: symmetric 42+42 sweep. A2/B2: finish legacy weighted sweep via shared JSON.",
    )
    parser.add_argument(
        "--part-a-completed",
        type=int,
        default=None,
        help=(
            "For --part A2/B2: progress hint only (e.g. 28 if host A stopped at 28/32). "
            "Actual work queue comes from legacy Part-A list minus valid JSON entries."
        ),
    )
    parser.add_argument(
        "--part-b-completed",
        type=int,
        default=None,
        help=(
            "For --part A2/B2: progress hint only (e.g. 32 if host B stopped at 32/53). "
            "Actual work queue comes from legacy Part-B list minus valid JSON entries."
        ),
    )
    parser.add_argument(
        "--n-k5-on-part-a",
        type=int,
        default=None,
        help=(
            "Legacy weighted Part-A size = n_blocks + this value (k=3 all blocks, then k=5 "
            "on blocks 0..N-1). Use 4 for a 32-task Part-A list (28+4); default formula gives 3 (31 tasks)."
        ),
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--adm-evaluator",
        type=Path,
        default=DEFAULT_ADM_EVALUATOR,
        help=f"ADM-style evaluator script (default: {DEFAULT_ADM_EVALUATOR})",
    )
    parser.add_argument(
        "--ref-batch",
        type=Path,
        default=DEFAULT_REF_BATCH,
        help=f"ADM reference .npz (default: {DEFAULT_REF_BATCH})",
    )
    parser.add_argument("--adm-python", type=str, default=sys.executable)

    parser.add_argument("--output-root", type=Path, default=Path("dit_s3cache/fid"))
    parser.add_argument("--results-json", type=Path, default=Path("dit_s3cache/fid/fid_sensitivity_results.json"))
    parser.add_argument("--gen-image-dir", type=Path, default=Path("dit_s3cache/fid/gen_image"))
    parser.add_argument("--sample-npz", type=Path, default=Path("dit_s3cache/fid/generated_samples.npz"))
    parser.add_argument("--keep-generated", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--block", type=int, default=None, help="Optional single block override.")
    parser.add_argument("--k", type=int, default=None, help="Optional single k override.")
    return parser


def main(args: argparse.Namespace) -> None:
    args.adm_evaluator = args.adm_evaluator.resolve()
    if not args.adm_evaluator.is_file():
        raise FileNotFoundError(
            f"ADM evaluator not found: {args.adm_evaluator}. "
            "Place evaluator.py under dit_s3cache/fid/ or pass --adm-evaluator."
        )

    args.ref_batch = args.ref_batch.resolve()
    if not args.ref_batch.is_file():
        raise FileNotFoundError(
            f"Reference batch not found: {args.ref_batch}. "
            "Place VIRTUAL_imagenet256_labeled.npz under dit_s3cache/fid/ or pass --ref-batch."
        )

    if args.model != "DiT-XL/2" or args.image_size != 256:
        raise ValueError("c_FID is currently scoped to DiT-XL/2 256x256.")
    if args.cfg_scale < 1.0:
        raise ValueError("cfg_scale should be >= 1.0 for DiT forward_with_cfg.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        raise RuntimeError("c_FID sampling is expected to run on CUDA.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.gen_image_dir.mkdir(parents=True, exist_ok=True)
    args.results_json.parent.mkdir(parents=True, exist_ok=True)

    model, diffusion, vae = load_everything(args, device)
    results = load_results(args)
    baseline_fid = ensure_baseline(model, diffusion, vae, args, results, device)
    if args.baseline_only:
        return

    tasks = get_task_list(
        args.part,
        args.k_values,
        n_blocks=len(model.blocks),
        part_a_completed=args.part_a_completed,
        part_b_completed=args.part_b_completed,
        n_k5_on_part_a_override=args.n_k5_on_part_a,
        results=results,
        args=args,
    )
    if args.block is not None:
        if args.k is None:
            raise ValueError("--block requires --k for a single experiment.")
        tasks = [(args.k, args.block)]
    elif args.k is not None:
        tasks = [(args.k, block_idx) for block_idx in range(len(model.blocks))]

    if args.part in ("A2", "B2"):
        n_k5_a = (
            args.n_k5_on_part_a
            if args.n_k5_on_part_a is not None
            else _legacy_n_k5_on_weighted_part_a(len(model.blocks))
        )
        a_list, b_list, pending = _legacy_remainder_pending(
            len(model.blocks), n_k5_a, results, args
        )
        _log_remainder_progress(
            a_list,
            b_list,
            pending,
            part_a_completed=args.part_a_completed,
            part_b_completed=args.part_b_completed,
        )
        print(
            f"Remainder phase {args.part}: {len(tasks)} tasks on this host "
            f"({len(pending)} pending in combined legacy pool)"
        )
    print(f"Part {args.part or 'FULL'}: {len(tasks)} cached evals + baseline")
    for task_idx, (k, block_idx) in enumerate(tasks, 1):
        results = load_results(args)
        baseline_fid = float(results["results"].get("baseline_fid", baseline_fid))
        if should_skip(results, args, k, block_idx):
            print(f"[{task_idx}/{len(tasks)}] skip existing (valid): k={k} block_{block_idx}")
            continue

        print(f"[{task_idx}/{len(tasks)}] evaluating k={k} block_{block_idx}")
        sampling_timesteps = list(reversed(diffusion.timestep_map))
        scheduler = create_dit_cache_config(
            target_block=block_idx,
            k=k,
            sampling_timesteps=sampling_timesteps,
            n_blocks=len(model.blocks),
        )
        run_result = generate_and_compute_fid(
            model=model,
            diffusion=diffusion,
            vae=vae,
            args=args,
            device=device,
            cache_scheduler=scheduler,
            experiment_name=f"k={k} block_{block_idx}",
            sampler=args.sampler,
            eta=args.eta,
        )
        fid_value = float(run_result["fid"])
        delta = fid_value - baseline_fid

        k_key = f"k{k}"
        results["results"].setdefault(k_key, {})
        results["results"][k_key][f"block_{block_idx}"] = {
            "fid": fid_value,
            "delta": delta,
            "cache_stats": run_result.get("cache_stats", {}),
            "sample_count": int(args.num_fid_samples),
        }
        save_results(results, args.results_json)
        print(f"k={k} block_{block_idx}: FID={fid_value:.6f}, delta={delta:+.6f}")


def load_everything(
    args: argparse.Namespace,
    device: str,
) -> tuple[torch.nn.Module, Any, AutoencoderKL]:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    model.load_state_dict(find_model(ckpt_path))
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    return model, diffusion, vae


def ensure_baseline(
    model: torch.nn.Module,
    diffusion: Any,
    vae: AutoencoderKL,
    args: argparse.Namespace,
    results: dict[str, Any],
    device: str,
) -> float:
    baseline_meta = results["results"].get("baseline_meta", {})
    baseline_fid = results["results"].get("baseline_fid")
    if baseline_fid is not None and baseline_matches(baseline_meta, args):
        print(f"Baseline exists: {float(baseline_fid):.6f}")
        return float(baseline_fid)

    if baseline_fid is not None:
        print("Existing baseline is incompatible with current config; recomputing.")
    baseline_result = generate_and_compute_fid(
        model=model,
        diffusion=diffusion,
        vae=vae,
        args=args,
        device=device,
        cache_scheduler=None,
        experiment_name="baseline",
        sampler=args.sampler,
        eta=args.eta,
    )
    results["results"]["baseline_fid"] = float(baseline_result["fid"])
    results["results"]["baseline_meta"] = baseline_result
    save_results(results, args.results_json)
    print(f"Baseline FID: {float(baseline_result['fid']):.6f}")
    return float(baseline_result["fid"])


def generate_and_compute_fid(
    *,
    model: torch.nn.Module,
    diffusion: Any,
    vae: AutoencoderKL,
    args: argparse.Namespace,
    device: str,
    cache_scheduler: dict[int, set[int]] | None,
    experiment_name: str,
    sampler: str = "ddpm",
    eta: float = 0.0,
) -> dict[str, Any]:
    clear_generated(args.gen_image_dir)
    args.sample_npz.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_npz.exists():
        args.sample_npz.unlink()

    cache_blocks = []
    sample_fn = model.forward_with_cfg
    if cache_scheduler is not None:
        cache_blocks = install_cache_wrappers(model, cache_scheduler)
        sample_fn = make_cached_forward_with_cfg(model, cache_blocks)

    try:
        generate_images(
            model=model,
            diffusion=diffusion,
            vae=vae,
            sample_fn=sample_fn,
            args=args,
            device=device,
            cache_blocks=cache_blocks,
            experiment_name=experiment_name,
            sampler=sampler,
            eta=eta,
        )
    finally:
        if cache_blocks:
            restore_cache_wrappers(cache_blocks)

    create_npz_from_sample_folder(
        args.gen_image_dir,
        args.sample_npz,
        args.num_fid_samples,
        experiment_name=experiment_name,
    )
    fid_value, evaluator_output = compute_fid_adm(
        adm_python=args.adm_python,
        adm_evaluator=args.adm_evaluator,
        ref_batch=args.ref_batch,
        sample_batch=args.sample_npz,
    )
    stats = cache_stats(cache_blocks) if cache_blocks else {}

    if not args.keep_generated:
        clear_generated(args.gen_image_dir)

    result = {
        "fid": fid_value,
        "num_fid_samples": int(args.num_fid_samples),
        "model": str(args.model),
        "image_size": int(args.image_size),
        "num_sampling_steps": int(args.num_sampling_steps),
        "sampler": str(sampler),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "vae": str(args.vae),
        "gen_image_dir": str(args.gen_image_dir),
        "sample_npz": str(args.sample_npz),
        "ref_batch": str(args.ref_batch),
        "adm_evaluator": str(args.adm_evaluator),
        "adm_output": evaluator_output,
        "cache_stats": stats,
    }
    if sampler == "ddim":
        result["eta"] = float(eta)
    return result


def generate_images(
    *,
    model: torch.nn.Module,
    diffusion: Any,
    vae: AutoencoderKL,
    sample_fn: Any,
    args: argparse.Namespace,
    device: str,
    cache_blocks: list[Any],
    experiment_name: str,
    sampler: str = "ddpm",
    eta: float = 0.0,
) -> None:
    sampler = str(sampler).lower()
    if sampler not in {"ddpm", "ddim"}:
        raise ValueError(f"sampler must be 'ddpm' or 'ddim', got {sampler!r}")
    seed_all(args.seed)
    latent_size = args.image_size // 8
    saved = 0
    total_batches = math.ceil(args.num_fid_samples / args.per_side_batch_size)
    iterator = range(total_batches)
    if args.progress:
        iterator = tqdm(iterator, desc=f"Generating [{experiment_name}]")

    for _ in iterator:
        current_bs = min(args.per_side_batch_size, args.num_fid_samples - saved)
        if current_bs <= 0:
            break

        if cache_blocks:
            reset_cache_state(cache_blocks)

        z, y = make_cfg_inputs(
            batch_size=current_bs,
            latent_size=latent_size,
            num_classes=args.num_classes,
            device=device,
        )
        loop_fn = diffusion.ddim_sample_loop if sampler == "ddim" else diffusion.p_sample_loop
        sample_kwargs = dict(
            model=sample_fn,
            shape=z.shape,
            noise=z,
            clip_denoised=False,
            model_kwargs={"y": y, "cfg_scale": args.cfg_scale},
            progress=False,
            device=device,
        )
        if sampler == "ddim":
            sample_kwargs["eta"] = float(eta)
        samples = loop_fn(**sample_kwargs)
        samples, _ = samples.chunk(2, dim=0)
        decoded = vae.decode(samples / 0.18215).sample
        decoded = (
            torch.clamp(127.5 * decoded + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        for sample_idx, sample in enumerate(decoded):
            Image.fromarray(sample).save(args.gen_image_dir / f"{saved + sample_idx:06d}.png")
        saved += decoded.shape[0]

        del z, y, samples, decoded
        torch.cuda.empty_cache()

    if saved != args.num_fid_samples:
        raise RuntimeError(f"Generated {saved} samples, expected {args.num_fid_samples}")


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


def create_npz_from_sample_folder(
    sample_dir: Path,
    npz_path: Path,
    num: int,
    *,
    experiment_name: str,
) -> Path:
    samples = []
    for idx in tqdm(range(num), desc=f"Building ADM sample npz [{experiment_name}]"):
        image_path = sample_dir / f"{idx:06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing generated image: {image_path}")
        samples.append(np.asarray(Image.open(image_path).convert("RGB")).astype(np.uint8))
    array = np.stack(samples)
    if array.shape != (num, 256, 256, 3):
        raise ValueError(f"Unexpected sample array shape for ADM evaluator: {array.shape}")
    np.savez(npz_path, arr_0=array)
    return npz_path


def compute_fid_adm(
    *,
    adm_python: str,
    adm_evaluator: Path,
    ref_batch: Path,
    sample_batch: Path,
) -> tuple[float, str]:
    # 必須 resolve：subprocess 的 cwd 設在 evaluator 目錄，相對路徑會錯解成
    # dit_s3cache/fid/dit_s3cache/fid/... 導致 FileNotFoundError。
    adm_evaluator = adm_evaluator.resolve()
    ref_batch = ref_batch.resolve()
    sample_batch = sample_batch.resolve()

    if not adm_evaluator.is_file():
        raise FileNotFoundError(f"ADM evaluator not found: {adm_evaluator}")
    if not ref_batch.is_file():
        raise FileNotFoundError(f"Reference batch not found: {ref_batch}")
    if not sample_batch.is_file():
        raise FileNotFoundError(f"Sample batch not found: {sample_batch}")

    command = [adm_python, str(adm_evaluator), str(ref_batch), str(sample_batch)]
    # evaluator.py 會把 Inception graph 下載到目前工作目錄；固定在其所在目錄避免污染 repo 根目錄。
    workdir = str(adm_evaluator.parent.resolve())
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        cwd=workdir,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    if completed.returncode != 0:
        raise RuntimeError(f"ADM evaluator failed with exit code {completed.returncode}:\n{output}")
    return parse_fid_from_adm_output(output), output


def parse_fid_from_adm_output(output: str) -> float:
    patterns = [
        r"\bFID\b\s*[:=]\s*([0-9eE+\-.]+)",
        r"\bfid\b\s*[:=]\s*([0-9eE+\-.]+)",
        r"\bfid\b\s+([0-9eE+\-.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    raise ValueError(f"Could not parse FID from ADM evaluator output:\n{output}")


def _legacy_n_k5_on_weighted_part_a(n_blocks: int) -> int:
    """k=5 block count on old weighted Part A (after all k=3). Default ~3 for n_blocks=28."""
    raw = (81 * n_blocks - 19) / 73.0
    c_a = int(round(raw))
    c_a = max(n_blocks, min(2 * n_blocks, c_a))
    return c_a - n_blocks


def legacy_weighted_part_a_tasks(n_blocks: int, n_k5_on_part_a: int) -> list[tuple[int, int]]:
    """Old Part-A order: all k=3, then k=5 on blocks ``0 .. n_k5_on_part_a - 1``."""
    if not 0 <= n_k5_on_part_a <= n_blocks:
        raise ValueError(f"n_k5_on_part_a must be in [0, {n_blocks}], got {n_k5_on_part_a}")
    return [(3, block_idx) for block_idx in range(n_blocks)] + [
        (5, block_idx) for block_idx in range(n_k5_on_part_a)
    ]


def legacy_weighted_part_b_tasks(n_blocks: int, n_k5_on_part_a: int) -> list[tuple[int, int]]:
    """Old Part-B order: k=5 on blocks ``[n_k5_on_part_a .. n_blocks)``, then all k=10."""
    if not 0 <= n_k5_on_part_a < n_blocks:
        raise ValueError(f"n_k5_on_part_a must be in [0, {n_blocks}), got {n_k5_on_part_a}")
    return [(5, block_idx) for block_idx in range(n_k5_on_part_a, n_blocks)] + [
        (10, block_idx) for block_idx in range(n_blocks)
    ]


def _legacy_remainder_pending(
    n_blocks: int,
    n_k5_on_part_a: int,
    results: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """All legacy A/B tasks that lack a valid ``fid`` entry in ``results``."""
    a_list = legacy_weighted_part_a_tasks(n_blocks, n_k5_on_part_a)
    b_list = legacy_weighted_part_b_tasks(n_blocks, n_k5_on_part_a)
    pending_a = [t for t in a_list if not should_skip(results, args, t[0], t[1])]
    pending_b = [t for t in b_list if not should_skip(results, args, t[0], t[1])]
    return a_list, b_list, pending_a + pending_b


def _log_remainder_progress(
    a_list: list[tuple[int, int]],
    b_list: list[tuple[int, int]],
    pending: list[tuple[int, int]],
    *,
    part_a_completed: int | None,
    part_b_completed: int | None,
) -> None:
    pending_a = [t for t in pending if t in a_list]
    pending_b = [t for t in pending if t in b_list]
    print(
        f"Legacy Part-A list: {len(a_list)} tasks "
        f"({len(pending_a)} pending in JSON); "
        f"Part-B list: {len(b_list)} tasks ({len(pending_b)} pending)"
    )
    if part_a_completed is not None:
        print(
            f"  Host A reported progress: {part_a_completed}/{len(a_list)} "
            f"(queue uses JSON, not only the tail {len(a_list) - part_a_completed} tasks)"
        )
    if part_b_completed is not None:
        print(
            f"  Host B reported progress: {part_b_completed}/{len(b_list)} "
            f"(queue uses JSON, not only the tail {len(b_list) - part_b_completed} tasks)"
        )
    if pending:
        preview = ", ".join(f"k{k}b{b}" for k, b in pending[:8])
        if len(pending) > 8:
            preview += ", ..."
        print(f"  Combined pending (first items): {preview}")


def get_task_list(
    part: str | None,
    k_values: list[int],
    n_blocks: int,
    *,
    part_a_completed: int | None = None,
    part_b_completed: int | None = None,
    n_k5_on_part_a_override: int | None = None,
    results: dict[str, Any] | None = None,
    args: argparse.Namespace | None = None,
) -> list[tuple[int, int]]:
    if part == "A":
        # Symmetric split (equal host speed): 42 + 42 when n_blocks=28.
        half = n_blocks // 2
        return [(3, block_idx) for block_idx in range(n_blocks)] + [
            (5, block_idx) for block_idx in range(half)
        ]

    if part == "B":
        half = n_blocks // 2
        return [(5, block_idx) for block_idx in range(half, n_blocks)] + [
            (10, block_idx) for block_idx in range(n_blocks)
        ]

    if part in ("A2", "B2"):
        if results is None or args is None:
            raise ValueError("Internal: A2/B2 requires results and args")
        n_k5_a = (
            n_k5_on_part_a_override
            if n_k5_on_part_a_override is not None
            else _legacy_n_k5_on_weighted_part_a(n_blocks)
        )
        _a_list, _b_list, pending = _legacy_remainder_pending(n_blocks, n_k5_a, results, args)
        if not pending:
            return []
        mid = (len(pending) + 1) // 2
        if part == "A2":
            return pending[:mid]
        return pending[mid:]

    return [(k, block_idx) for k in k_values for block_idx in range(n_blocks)]


def load_results(args: argparse.Namespace) -> dict[str, Any]:
    default_config = {
        "model": args.model,
        "image_size": int(args.image_size),
        "num_sampling_steps": int(args.num_sampling_steps),
        "sampler": str(args.sampler),
        "cfg_scale": float(args.cfg_scale),
        "eval_samples": int(args.num_fid_samples),
        "fid_method": "ADM evaluator",
        "adm_evaluator": str(args.adm_evaluator),
        "ref_batch": str(args.ref_batch),
        "seed": int(args.seed),
        "k_values": list(args.k_values),
    }
    if args.sampler == "ddim":
        default_config["eta"] = float(args.eta)
    if args.results_json.exists():
        with open(args.results_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("config", default_config)
        payload.setdefault("results", {})
        return payload
    return {"config": default_config, "results": {}}


def save_results(results: dict[str, Any], path: Path) -> None:
    results["last_updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def should_skip(
    results: dict[str, Any],
    args: argparse.Namespace,
    k: int,
    block_idx: int,
) -> bool:
    """Skip if JSON has a *complete* entry for this (k, block)."""
    entry = results.get("results", {}).get(f"k{k}", {}).get(f"block_{block_idx}")
    if entry is None:
        return False
    if not isinstance(entry, dict):
        return False
    if "fid" not in entry:
        return False
    try:
        float(entry["fid"])
    except (TypeError, ValueError):
        return False
    if entry.get("sample_count") is not None and int(entry["sample_count"]) != int(
        args.num_fid_samples
    ):
        return False
    return True


def baseline_matches(meta: dict[str, Any], args: argparse.Namespace) -> bool:
    if not isinstance(meta, dict):
        return False
    return (
        int(meta.get("num_fid_samples", -1)) == int(args.num_fid_samples)
        and int(meta.get("num_sampling_steps", -1)) == int(args.num_sampling_steps)
        and str(meta.get("sampler", "ddpm")) == str(args.sampler)
        and (
            str(args.sampler) != "ddim"
            or float(meta.get("eta", float("nan"))) == float(args.eta)
        )
        and int(meta.get("seed", -1)) == int(args.seed)
        and str(meta.get("model", "")) == str(args.model)
        and int(meta.get("image_size", -1)) == int(args.image_size)
        and float(meta.get("cfg_scale", -1.0)) == float(args.cfg_scale)
        and str(meta.get("vae", "")) == str(args.vae)
        and str(meta.get("ref_batch", "")) == str(args.ref_batch)
        and str(meta.get("adm_evaluator", "")) == str(args.adm_evaluator)
    )


def clear_generated(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main(build_argparser().parse_args())
