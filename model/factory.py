"""One construction path for training, evaluation, and inference."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dataset.manifest import sha256_file
from dataset.motion_representation import MotionNormalizer, load_geometry
from model.conditioning import MultiModalConditioner
from model.gaussian_diffusion import GaussianDiffusion
from model.skeleton_mamba import SkeletonMamba
from utils.config import validate_config
from utils.kinematics import rest_offsets
from utils.precision import amp_dtype, autocast_context, check_cuda_precision


class MotionSystem(nn.Module):
    def __init__(self, denoiser, conditioner, diffusion, normalizer, geometry_offsets=None, *, sampling_amp_dtype=None):
        super().__init__()
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.feature_contract = None
        self.sampling_amp_dtype = sampling_amp_dtype
        # Fixed buffers let checkpoints restore geometry without source assets.
        # has_geometry disables the zero-initialized placeholders.
        offsets = torch.zeros(24, 3) if geometry_offsets is None else torch.as_tensor(geometry_offsets).float()
        if offsets.shape != (24, 3) or not torch.isfinite(offsets).all():
            raise ValueError("geometry offsets must be finite [24,3]")
        self.register_buffer("geometry_offsets", offsets.clone())
        self.register_buffer("has_geometry", torch.tensor(geometry_offsets is not None, dtype=torch.bool))

    def forward(self, batch, *, t=None, noise=None):
        conditions = self.conditioner(batch["audio"], batch["video"], batch["mask"])
        return self.diffusion.training_losses(
            self.denoiser, batch["motion"], conditions, batch["mask"],
            normalizer=self.normalizer, t=t, noise=noise,
            contacts=batch.get("contacts"),
            offsets=batch.get("offsets", self.geometry_offsets if bool(self.has_geometry) else None),
        )

    @torch.no_grad()
    def sample(self, audio, video, mask, *, head_guidance=None, generator=None, initial_noise=None):
        """Generate without accepting target motion or target-derived world transforms."""
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            enabled = audio.is_cuda and self.sampling_amp_dtype is not None
            if enabled:
                check_cuda_precision(audio.device, self.sampling_amp_dtype)
            with autocast_context(audio.device, enabled=enabled, dtype=self.sampling_amp_dtype):
                conditions = self.conditioner(audio, video, mask)
                mask = conditions["context_mask"]
                normalized = self.diffusion.sample(
                    self.denoiser, (*audio.shape[:2], 24, 9), conditions, mask,
                    normalizer=self.normalizer, head_guidance=head_guidance,
                    generator=generator, initial_noise=initial_noise,
                )
            return self.normalizer.denormalize(normalized) * mask[..., None, None]
        finally:
            for module, training in modes:
                module.training = training


def build_system(config: dict, *, load_statistics: bool = True) -> MotionSystem:
    config = validate_config(config)
    normalizer = (MotionNormalizer.from_npz(config["data"]["statistics"])
                  if load_statistics else MotionNormalizer(torch.zeros(24, 9), torch.ones(24, 9)))
    offsets = None
    if load_statistics:
        geometry_path = config["data"].get("geometry")
        if geometry_path:
            joints, parents, _ = load_geometry(geometry_path)
            if sha256_file(geometry_path) != normalizer.metadata.get("geometry_hash"):
                raise ValueError("data.geometry does not match the normalization/motion geometry hash")
            offsets = rest_offsets(joints, parents)
        requires_geometry = any(config["diffusion"].get("loss_weights", {}).get(key, 0)
                                for key in ("position", "consistency", "contact"))
        if requires_geometry and offsets is None:
            raise ValueError("enabled FK/contact losses require data.geometry matching prepared motion")
    return MotionSystem(
        SkeletonMamba(**config["model"]), MultiModalConditioner(**config["conditioning"]),
        GaussianDiffusion(**config["diffusion"]), normalizer, offsets,
        sampling_amp_dtype=amp_dtype(config["training"]) if config["training"].get("amp", False) else None,
    )


def load_system_checkpoint(path: str | Path, device="cpu", *, use_ema=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1 or "config" not in checkpoint:
        raise ValueError("Not a versioned SkeletonMamba reconstruction checkpoint")
    if checkpoint["config"]["model"].get("backend", "reference") == "mamba2" and torch.device(device).type != "cuda":
        raise RuntimeError("This checkpoint uses fused Mamba2; load it on a CUDA device")
    system = build_system(checkpoint["config"], load_statistics=False)
    key = "ema" if use_ema else "model"
    if key not in checkpoint:
        raise ValueError(f"Checkpoint has no {key} state")
    system.load_state_dict(checkpoint[key], strict=True)
    if (not torch.isfinite(system.normalizer.mean).all()
            or not torch.isfinite(system.normalizer.std).all()
            or not (system.normalizer.std > 0).all()):
        raise ValueError("checkpoint contains invalid normalization buffers: expected finite mean and positive std")
    system.normalizer.metadata = checkpoint.get("normalization_metadata", {})
    system.feature_contract = checkpoint.get("feature_contract")
    if not torch.isfinite(system.geometry_offsets).all():
        raise ValueError("checkpoint contains nonfinite geometry offsets")
    requires_geometry = any(checkpoint["config"]["diffusion"].get("loss_weights", {}).get(name, 0)
                            for name in ("position", "consistency", "contact"))
    if requires_geometry and not bool(system.has_geometry):
        raise ValueError("checkpoint declares FK losses without embedded geometry")
    return system.to(device).eval(), checkpoint
