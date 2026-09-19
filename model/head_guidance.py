"""Head guidance from estimated input-video trajectories, without target motion.
"""
import math

import torch

from model.losses import denormalize, decoded_positions, validate_mask
from utils.kinematics import matrix_to_axis_angle, rotation_6d_to_matrix


GUIDANCE_PROTOCOLS = ('posterior_log_v1', 'legacy_clean_frobenius_v1')


def so3_log_frobenius_squared(rotation, target_rotation):
    """||log(R R_target^T)||_F² = 2 theta², without acos endpoint singularities.

    The quaternion/axis-angle conversion uses stable identity and near-pi
    branches. At exactly pi the derivative is non-unique; the conversion
    selects a finite branch.
    """
    with torch.autocast(device_type=rotation.device.type, enabled=False):
        dtype = torch.float64 if rotation.dtype == torch.float64 else torch.float32
        relative = rotation.to(dtype) @ target_rotation.to(dtype).transpose(-1, -2)
        axis_angle = matrix_to_axis_angle(relative)
        return 2 * axis_angle.square().sum(-1)


class EstimatedHeadGuidance:
    def __init__(self, positions, rotations=None, confidence=None, mask=None, *,
                 strength=0.01, position_weight=1.0, rotation_weight=1.0,
                 coordinate_frame=None, head_index=15, offsets=None, parents=None,
                 protocol='posterior_log_v1'):
        if protocol not in GUIDANCE_PROTOCOLS:
            raise ValueError(f'guidance protocol must be one of {GUIDANCE_PROTOCOLS}')
        self.protocol = protocol
        if not coordinate_frame or not isinstance(coordinate_frame, str):
            raise ValueError("estimated heads require a declared coordinate_frame")
        for name, value in (("strength", strength), ("position_weight", position_weight),
                            ("rotation_weight", rotation_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.positions = torch.as_tensor(positions).detach()
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError("estimated positions must be [B,T,3]")
        self.rotations = None if rotations is None else torch.as_tensor(rotations).detach()
        if self.rotations is not None and self.rotations.shape != (*self.positions.shape[:2], 3, 3):
            raise ValueError("estimated rotations must be [B,T,3,3]")
        self.confidence = None if confidence is None else torch.as_tensor(confidence).detach()
        if self.confidence is not None:
            if self.confidence.shape != self.positions.shape[:2]:
                raise ValueError("confidence must be [B,T]")
            if not torch.isfinite(self.confidence).all() or (self.confidence < 0).any() or (self.confidence > 1).any():
                raise ValueError("confidence must be finite in [0,1]")
        self.mask = None if mask is None else torch.as_tensor(mask, dtype=torch.bool)
        if self.mask is not None and self.mask.shape != self.positions.shape[:2]:
            raise ValueError("head mask must be [B,T]")
        self.strength, self.position_weight, self.rotation_weight = strength, position_weight, rotation_weight
        self.coordinate_frame, self.head_index = coordinate_frame, head_index
        self.offsets, self.parents = offsets, parents

    def energy_per_sequence(self, state, mask=None, normalizer=None):
        """Independent [B] energies; confidence is an explicit non-paper extension.

        The default averages confidence-weighted costs over valid frames, so
        lower confidence reduces the correction. The legacy protocol preserves
        its confidence-sum denominator separately within each sequence.
        """
        with torch.autocast(device_type=state.device.type, enabled=False):
            state = state.to(dtype=torch.float64 if state.dtype == torch.float64 else torch.float32)
            return self._energy_per_sequence(state, mask, normalizer)

    def _energy_per_sequence(self, state, mask=None, normalizer=None):
        valid = validate_mask(mask, state)
        if state.shape[:2] != self.positions.shape[:2]:
            raise ValueError("head estimates and motion must share physical timestamps and shape")
        if self.mask is not None:
            valid = valid & self.mask.to(valid.device)
        weights = valid.to(state.dtype)
        if self.confidence is not None:
            weights = weights * self.confidence.to(state)
        active = weights > 0
        # Sanitize excluded state values before decoding. Masking NaNs only
        # after a rotation/FK operation can still contaminate backward passes.
        safe_state = torch.where(active[..., None, None], state, torch.zeros_like(state))
        raw = denormalize(safe_state, normalizer)
        xyz = (raw[..., self.head_index, :3] if self.offsets is None else
               decoded_positions(raw, self.offsets, self.parents)[..., self.head_index, :])
        target = self.positions.to(raw)
        if not torch.isfinite(target[active]).all():
            raise ValueError("valid estimated head positions must be finite")
        target = torch.where(active[..., None], target, torch.zeros_like(target))
        xyz = torch.where(active[..., None], xyz, torch.zeros_like(xyz))
        energy = self.position_weight * (xyz - target).square().sum(-1)
        if self.rotations is not None and self.rotation_weight:
            target_rotation = self.rotations.to(raw)
            if not torch.isfinite(target_rotation[active]).all():
                raise ValueError("valid estimated head rotations must be finite")
            identity = torch.eye(3, dtype=raw.dtype, device=raw.device)
            target_rotation = torch.where(active[..., None, None], target_rotation, identity)
            rotation = rotation_6d_to_matrix(raw[..., self.head_index, 3:])
            rotation = torch.where(active[..., None, None], rotation, identity)
            if self.protocol == 'posterior_log_v1':
                rotation_energy = so3_log_frobenius_squared(rotation, target_rotation)
            else:
                rotation_energy = (rotation - target_rotation).square().sum((-1, -2))
            energy = energy + self.rotation_weight * rotation_energy
        denominator = (valid.sum(-1) if self.protocol == 'posterior_log_v1' else weights.sum(-1)).clamp_min(1)
        return (energy * weights).sum(-1) / denominator

    def energy(self, state, mask=None, normalizer=None):
        """Mean energy for diagnostics; correction gradients never use this mean."""
        return self.energy_per_sequence(state, mask, normalizer).mean()

    def _gradient(self, state, mask, normalizer):
        with torch.enable_grad(), torch.autocast(device_type=state.device.type, enabled=False):
        # Rotation geometry requires at least float32; preserve float64 inputs.
            differentiable = state.detach().to(dtype=torch.float64 if state.dtype == torch.float64 else torch.float32)
            differentiable.requires_grad_(True)
            energies = self.energy_per_sequence(differentiable, mask, normalizer)
            gradient, = torch.autograd.grad(energies.sum(), differentiable)
        return gradient.to(state.dtype)

    def guide_posterior(self, mean, variance, mask=None, normalizer=None, timestep=None):
        """Return mu - lambda Sigma grad_mu E independently for each sequence."""
        if self.protocol != 'posterior_log_v1':
            raise ValueError('guide_posterior requires posterior_log_v1')
        variance = torch.as_tensor(variance, device=mean.device, dtype=mean.dtype)
        if not torch.isfinite(variance).all() or (variance < 0).any():
            raise ValueError('posterior variance must be finite and nonnegative')
        try:
            variance = torch.broadcast_to(variance, mean.shape)
        except RuntimeError as exc:
            raise ValueError('posterior variance must broadcast to motion state') from exc
        if self.strength == 0 or not variance.any():
            return mean
        gradient = self._gradient(mean, mask, normalizer)
        return (mean.detach() - self.strength * variance * gradient).detach()

    def __call__(self, state, mask=None, normalizer=None, timestep=None, *, variance=None):
        if self.strength == 0:
            return state
        if self.protocol == 'posterior_log_v1':
            if variance is None:
                raise ValueError('posterior_log_v1 requires posterior variance; use guide_posterior')
            return self.guide_posterior(state, variance, mask, normalizer, timestep)
        gradient = self._gradient(state, mask, normalizer)
        return (state.detach() - self.strength * gradient).detach()
