"""Configuration shared by training, evaluation and condition-only inference."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return validate_config(config)


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected configuration schema_version: 1")
    config = copy.deepcopy(config)
    required = {"model", "conditioning", "diffusion", "data", "training"}
    if required - config.keys():
        raise ValueError(f"Missing configuration sections: {sorted(required - config.keys())}")
    for name in required:
        if not isinstance(config[name], dict):
            raise ValueError(f"{name} must be a mapping")
    if config["model"].get("input_dim", 9) != 9 or config["model"].get("num_joints", 24) != 24:
        raise ValueError("The supported motion representation is SMPL24 with xyz + rotation6d")
    if config["model"].get("d_context", 256) != config["conditioning"].get("context_dim", 256):
        raise ValueError("Model and conditioning context widths differ")
    window = config["data"].get("window", 150)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("data.window must be positive")
    training = config["training"]
    if not isinstance(training.get("amp", False), bool):
        raise ValueError("training.amp must be a boolean")
    if training.get("amp_dtype", "float16") not in ("float16", "bfloat16"):
        raise ValueError("training.amp_dtype must be float16 or bfloat16")
    for name in ("batch_size", "max_steps", "accumulation_steps", "save_every", "validate_every", "validation_batches", "log_every", "cpu_threads"):
        value = training.get(name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{name} must be positive")
    if training.get("num_workers", 0) != 0:
        raise ValueError("The resumable reference trainer requires num_workers=0 (no prefetched crops)")
    if not 0 <= training.get("ema_decay", 0.995) < 1:
        raise ValueError("ema_decay must be in [0,1)")
    for name, default in (("learning_rate", 2e-4), ("gradient_clip", 1.0)):
        value = training.get(name, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{name} must be finite and positive")
    decay = training.get("weight_decay", 0.01)
    if not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:
        raise ValueError("training.weight_decay must be finite and nonnegative")
    betas = training.get("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
        raise ValueError("AdamW betas must be two values in [0,1)")
    if config["model"].get("backend", "reference") not in ("reference", "mamba2"):
        raise ValueError("Select the reference or explicitly installed mamba2 backend")
    if config["conditioning"].get("fusion_architecture", "legacy_cross_attention_v1") not in (
            "legacy_cross_attention_v1", "paper_encoder_v1"):
        raise ValueError("Unknown conditioning.fusion_architecture")
    if config["diffusion"].get("loss_protocol", "legacy_xyz_v1") not in ("legacy_xyz_v1", "paper_v1"):
        raise ValueError("Unknown diffusion.loss_protocol")
    return config


def model_protocols(config: dict) -> dict:
    """Report effective definitions, including defaults for historical checkpoints."""
    return {"fusion_architecture": config["conditioning"].get("fusion_architecture", "legacy_cross_attention_v1"),
            "loss_protocol": config["diffusion"].get("loss_protocol", "legacy_xyz_v1")}


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
