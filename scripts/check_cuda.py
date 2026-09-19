"""Profile fused Mamba2 training on synthetic inputs.

Includes forward/backward, AdamW, gradient accumulation, and EMA. Excludes data
loading, preprocessing, validation, and checkpoint I/O.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from model.factory import build_system
from trainer import optimizer_groups
from utils.config import load_config, validate_config
from utils.kinematics import (axis_angle_to_matrix, global_rotations_to_positions,
                              matrix_to_rotation_6d)
from utils.precision import amp_dtype, autocast_context, check_cuda_precision


def default_config():
    """One full-width layer exercises group (768), joint, and temporal scans."""
    return {
        "schema_version": 1,
        "model": {"input_dim": 9, "embed_dim": 128, "depth": 1,
                  "num_joints": 24, "d_context": 256, "backend": "mamba2",
                  "d_state": 64, "headdim": 64, "expand": 2},
        "conditioning": {"fusion_architecture": "paper_encoder_v1",
                         "audio_dim": 4800, "video_dim": 2048,
                         "context_dim": 256, "hidden_dim": 512,
                         "num_layers": 2, "num_heads": 4, "dropout": 0.0},
        "diffusion": {"loss_protocol": "paper_v1", "timesteps": 1000,
                      "objective": "pred_x0", "beta_schedule": "cosine",
                      "loss_type": "l1", "fps": 30,
                      "loss_weights": {"position": 1.0, "velocity": 0.1,
                                       "contact": 1.0, "alignment": 0.01}},
        "data": {"window": 150},
        "training": {"device": "cuda", "batch_size": 1, "accumulation_steps": 1,
                     "learning_rate": 2e-4, "weight_decay": 0.01,
                     "gradient_clip": 1.0, "ema_decay": 0.995,
                     "amp": True, "amp_dtype": "bfloat16",
                     "seed": 42, "deterministic": False, "cpu_threads": 4},
    }


def prepare_config(config, *, batch_size=None, window=None, precision=None):
    config = copy.deepcopy(config)
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
    if window is not None:
        config["data"]["window"] = window
    if precision is not None:
        if precision not in ("bf16", "fp16", "fp32"):
            raise ValueError("precision must be bf16, fp16, or fp32")
        config["training"]["amp"] = precision != "fp32"
        config["training"]["amp_dtype"] = "bfloat16" if precision == "bf16" else "float16"
    config = validate_config(config)
    if config["model"].get("backend") != "mamba2":
        raise ValueError("CUDA preflight requires model.backend: mamba2; select a CUDA config")
    return config


def package_info(distribution, module):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    try:
        spec = importlib.util.find_spec(module)
        path = None if spec is None else spec.origin
    except (ImportError, ValueError, AttributeError):
        path = None
    return {"version": version, "module_path": path}


def environment_report(device="cuda"):
    result = {"host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "python": platform.python_version(), "torch": str(torch.__version__),
              "torch_path": torch.__file__, "torch_cuda_build": torch.version.cuda,
              "triton_f32_default_override": os.environ.get("TRITON_F32_DEFAULT"),
              "cuda_available": torch.cuda.is_available(), "requested_device": str(device),
              "packages": {"mamba_ssm": package_info("mamba-ssm", "mamba_ssm"),
                           "causal_conv1d": package_info("causal-conv1d", "causal_conv1d"),
                           "triton": package_info("triton", "triton")}}
    try:
        result["driver"] = Path("/proc/driver/nvidia/version").read_text().strip()
    except OSError:
        result["driver"] = None
    requested = torch.device(device)
    if result["cuda_available"] and requested.type == "cuda":
        props = torch.cuda.get_device_properties(requested)
        with torch.cuda.device(requested):
            bf16 = torch.cuda.is_bf16_supported()
        result["gpu"] = {"name": props.name, "total_vram_bytes": props.total_memory,
                         "total_vram_gib": props.total_memory / 2**30,
                         "compute_capability": [props.major, props.minor],
                         "bfloat16_supported": bf16}
    return result


def require_cuda(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("This preflight requires --device cuda or cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a CUDA-enabled PyTorch installation "
                           "on an allocated NVIDIA GPU; this command has no CPU fallback.")
    torch.cuda.get_device_properties(device)  # Validate the selected ordinal.
    return device


def synthetic_batch(config, device):
    """Build synthetic motion and features using arbitrary, non-SMPL geometry."""
    batch = config["training"]["batch_size"]
    frames = config["data"].get("window", 150)
    phase = torch.linspace(0, 2, frames, device=device)
    joints = torch.arange(24, device=device, dtype=torch.float32)
    offsets = torch.stack((0.03 * torch.sin(joints), 0.03 * torch.cos(joints),
                           torch.full_like(joints, 0.08)), -1)
    offsets[0] = 0
    angles = torch.zeros(batch, frames, 24, 3, device=device)
    angles[..., 2] = 0.1 * torch.sin(phase[:, None] + joints[None, :] * 0.1)
    rotations = axis_angle_to_matrix(angles)
    root = torch.zeros(batch, frames, 3, device=device)
    root[..., 0] = 0.05 * torch.sin(phase)
    positions = global_rotations_to_positions(rotations, root, offsets)
    motion = torch.cat((positions, matrix_to_rotation_6d(rotations)), -1)
    conditioning = config["conditioning"]
    contacts = ((torch.arange(frames, device=device)[:, None] +
                 torch.arange(4, device=device)[None, :]) % 3 != 0)
    return {"motion": motion, "mask": torch.ones(batch, frames, device=device, dtype=torch.bool),
            "audio": torch.randn(batch, frames, conditioning["audio_dim"], device=device),
            "video": torch.randn(batch, frames, conditioning["video_dim"], device=device),
            "contacts": contacts[None].expand(batch, -1, -1), "offsets": offsets}


def run_probe(config, *, device="cuda", steps=3, warmup=1):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    config = prepare_config(config)
    device = require_cuda(device)
    options = config["training"]
    enabled = bool(options.get("amp", False))
    dtype = amp_dtype(options)
    if enabled:
        check_cuda_precision(device, dtype)
    torch.set_num_threads(options.get("cpu_threads", 4))
    torch.manual_seed(options.get("seed", 42))
    torch.cuda.manual_seed_all(options.get("seed", 42))
    torch.use_deterministic_algorithms(options.get("deterministic", False))
    torch.backends.cudnn.benchmark = False
    model = build_system(config, load_statistics=False).to(device).train()
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(optimizer_groups(model, options.get("weight_decay", 0.01)),
                                  lr=options.get("learning_rate", 2e-4),
                                  betas=tuple(options.get("betas", [0.9, 0.999])))
    scaler = torch.amp.GradScaler("cuda", enabled=enabled and dtype == torch.float16)
    batch = synthetic_batch(config, device)
    accumulation = options.get("accumulation_steps", 1)

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accumulation):
            with autocast_context(device, enabled=enabled, dtype=dtype):
                loss = model(batch)["loss"] / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss in CUDA preflight")
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.get("gradient_clip", 1.0),
                                       error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            source = model.state_dict()
            for name, tensor in ema.state_dict().items():
                if tensor.is_floating_point():
                    tensor.lerp_(source[name], 1 - options.get("ema_decay", 0.995))
                else:
                    tensor.copy_(source[name])
        return loss_sum

    for _ in range(warmup):
        update()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings, losses = [], []
    for _ in range(steps):
        start = time.perf_counter()
        losses.append(update())
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    gradient_checks = {}
    for name in ("denoiser", "conditioner"):
        gradients = [p.grad.detach().float().norm() for p in getattr(model, name).parameters()
                     if p.grad is not None]
        if not gradients:
            raise RuntimeError(f"No {name} gradients were produced")
        norm = torch.linalg.vector_norm(torch.stack(gradients))
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Invalid or zero {name} gradient")
        gradient_checks[name] = float(norm)
    median = statistics.median(timings)
    return {"status": "passed", "environment": environment_report(device),
            "synthetic": True,
            "scope": "training forward/backward, accumulation, AdamW, clipping, and GPU EMA; "
                     "excludes data loading, preprocessing, validation, and checkpoint I/O",
            "geometry": "arbitrary synthetic offsets; not SMPL assets",
            "model": config["model"], "conditioning": config["conditioning"],
            "diffusion": config["diffusion"], "normalization": "identity synthetic statistics",
            "precision": str(dtype).removeprefix("torch.") if enabled else "float32",
            "microbatch_size": options["batch_size"], "accumulation_steps": accumulation,
            "effective_batch_size": options["batch_size"] * accumulation,
            "frames": config["data"].get("window", 150), "warmup_updates": warmup,
            "measured_updates": steps, "parameters": sum(p.numel() for p in model.parameters()),
            "seconds_per_update": timings, "median_seconds_per_update": median,
            "examples_per_second": options["batch_size"] * accumulation / median,
            "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_vram_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "losses": losses, "gradient_norms": gradient_checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="CUDA YAML config; default uses one full-width layer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3, help="Measured optimizer updates")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed optimizer updates")
    parser.add_argument("--batch-size", type=int, help="Override microbatch size")
    parser.add_argument("--window", type=int, help="Override frame count")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--json", type=Path, help="Also save the diagnostic report")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config) if args.config else default_config()
        config = prepare_config(config, batch_size=args.batch_size, window=args.window,
                                precision=args.amp_dtype)
        report = run_probe(config, device=args.device, steps=args.steps, warmup=args.warmup)
    except (RuntimeError, ValueError, ImportError, OSError, AssertionError) as exc:
        causes = [str(exc)]
        cause = exc.__cause__
        while cause is not None:
            causes.append(str(cause))
            cause = cause.__cause__
        try:
            environment = environment_report(args.device)
        except (RuntimeError, ValueError, AssertionError):
            environment = environment_report("cuda") if not torch.cuda.is_available() else {}
        report = {"status": "failed", "errors": causes, "environment": environment}
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text, file=sys.stdout if report["status"] == "passed" else sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
