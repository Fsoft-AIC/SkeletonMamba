"""Masked losses for the explicit global xyz/global rotation6d contract.
"""
from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F

from utils.kinematics import rotation_6d_to_matrix, global_rotations_to_positions, SMPL_PARENTS


def validate_mask(mask, motion):
    if mask is None:
        return torch.ones(motion.shape[:2], dtype=torch.bool, device=motion.device)
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(motion.shape[:2]):
        raise ValueError("mask must be boolean [B,T]")
    mask = mask.to(motion.device)
    if not mask.any(dim=1).all():
        raise ValueError("each example must contain at least one valid frame")
    return mask


def masked_mean(value, mask):
    """Valid-element mean per sequence, then equal-weight batch mean.

    Empty selections (e.g. a sequence without contact transitions) contribute
    differentiable zero. Sequence weighting makes example-weighted microbatch
    accumulation equivalent to an effective batch even for unequal lengths.
    """
    mask = mask.to(device=value.device, dtype=torch.bool)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    selected = torch.where(mask, value, torch.zeros_like(value)).flatten(start_dim=1)
    counts = mask.flatten(start_dim=1).sum(dim=1)
    return (selected.sum(dim=1) / counts.clamp_min(1)).mean()


def masked_pose_sum_mean(value, mask):
    """Sum pose features, average valid frames, then average sequences.

    This is the squared-vector-norm reduction in supplementary Eqs. 26--28.
    The temporal denominator includes every valid frame/transition, even when
    the numerator is zero because no foot is in contact.
    """
    return masked_mean(value.flatten(start_dim=2).sum(dim=-1), mask)


def denormalize(motion, normalizer=None):
    if normalizer is None:
        return motion
    if hasattr(normalizer, "denormalize"):
        return normalizer.denormalize(motion)
    if isinstance(normalizer, Mapping):
        mean, std = normalizer["mean"], normalizer["std"]
    elif isinstance(normalizer, (tuple, list)) and len(normalizer) >= 2:
        mean, std = normalizer[:2]
    else:
        raise TypeError("normalizer must provide denormalize or mean/std")
    mean = torch.as_tensor(mean, device=motion.device, dtype=motion.dtype)
    std = torch.as_tensor(std, device=motion.device, dtype=motion.dtype)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("normalization statistics must be finite with positive std")
    return motion * std + mean


def decoded_positions(motion, offsets, parents=None):
    if offsets is None:
        raise ValueError("FK geometry requires explicit rest offsets")
    offsets = torch.as_tensor(offsets, device=motion.device, dtype=motion.dtype)
    return global_rotations_to_positions(
        rotation_6d_to_matrix(motion[..., 3:]), motion[..., 0, :3], offsets,
        SMPL_PARENTS if parents is None else parents,
    )


def symmetric_alignment_loss(audio, video, mask, temperature=0.1):
    """Symmetric temporal InfoNCE, per-sequence negatives and equal sequence weights."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("alignment temperature must be positive")
    if audio.shape != video.shape or audio.ndim != 3 or audio.shape[:2] != mask.shape or mask.dtype != torch.bool:
        raise ValueError("aligned audio/video embeddings must share [B,T,C] and mask [B,T]")
    losses = []
    for a, v, valid in zip(audio, video, mask):
        a, v = a[valid].float(), v[valid].float()
        if a.shape[0] == 0:
            losses.append((a.sum() + v.sum()) * 0)
            continue
        logits = F.normalize(a, dim=-1) @ F.normalize(v, dim=-1).T / temperature
        targets = torch.arange(a.shape[0], device=a.device)
        losses.append((F.cross_entropy(logits, targets) +
                       F.cross_entropy(logits.T, targets)) / 2)
    return torch.stack(losses).mean()


def geometry_losses(prediction, target, mask, *, normalizer=None, weights=None,
                    offsets=None, parents=None, contacts=None, fps=30,
                    foot_indices=(7, 8, 10, 11), loss_protocol="legacy_xyz_v1"):
    """Return only explicitly enabled terms. Position/consistency require FK assets.

    ``paper_v1`` follows supplementary Eqs. 26--28: position sums squared FK
    coordinate errors per frame; velocity sums full-pose frame-difference
    errors; contact sums gated FK foot displacements over all valid transitions.
    Interpreting the paper's pose vector as this repository's denormalized
    global xyz + rotation6d representation is an explicit reconstruction choice.
    Frame contact labels [B,T,F] gate the transition starting at that frame.

    ``legacy_xyz_v1`` retains xyz velocity in m/s, element-mean position errors,
    and contact-element-mean foot velocity in m/s with both endpoint labels
    required. Both protocols also accept explicit [B,T-1,F] transition labels.
    Consistency is an optional non-paper term with its existing element mean.
    """
    if loss_protocol not in ("legacy_xyz_v1", "paper_v1"):
        raise ValueError("loss_protocol must be legacy_xyz_v1 or paper_v1")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    weights = weights or {}
    needed = any(weights.get(k, 0) for k in ("position", "consistency", "velocity", "contact"))
    zero = torch.where(mask[..., None, None], prediction, torch.zeros_like(prediction)).sum() * 0
    result = dict.fromkeys(("position", "consistency", "velocity", "contact"), zero)
    if not needed:
        return result
    pred = denormalize(prediction.float(), normalizer)
    gt = denormalize(target.float(), normalizer)
    pred = torch.where(mask[..., None, None], pred, torch.zeros_like(pred))
    gt = torch.where(mask[..., None, None], gt, torch.zeros_like(gt))
    fk_pred = None
    if any(weights.get(k, 0) for k in ("position", "consistency", "contact")):
        fk_pred = decoded_positions(pred, offsets, parents)
    if weights.get("position", 0):
        fk_target = decoded_positions(gt, offsets, parents)
        reduce_position = masked_pose_sum_mean if loss_protocol == "paper_v1" else masked_mean
        result["position"] = reduce_position((fk_pred - fk_target).square(), mask)
    if weights.get("consistency", 0):
        result["consistency"] = masked_mean((pred[..., :3] - fk_pred).square(), mask)
    pair_mask = mask[:, 1:] & mask[:, :-1]
    if weights.get("velocity", 0):
        if loss_protocol == "paper_v1":
            difference = torch.diff(pred, dim=1) - torch.diff(gt, dim=1)
            result["velocity"] = masked_pose_sum_mean(difference.square(), pair_mask)
        else:
            pred_vel = torch.diff(pred[..., :3], dim=1) * fps
            target_vel = torch.diff(gt[..., :3], dim=1) * fps
            result["velocity"] = masked_mean((pred_vel - target_vel).square(), pair_mask)
    if weights.get("contact", 0):
        if contacts is None:
            raise ValueError("contact loss requires explicit contact labels")
        labels = torch.as_tensor(contacts, device=prediction.device)
        if not torch.isfinite(labels).all() or ((labels != 0) & (labels != 1)).any():
            raise ValueError("contacts must be finite binary labels")
        expected = (prediction.shape[0], prediction.shape[1], len(foot_indices))
        if labels.shape == expected:
            labels = (labels[:, :-1].bool() if loss_protocol == "paper_v1" else
                      labels[:, :-1].bool() & labels[:, 1:].bool())
        elif labels.shape == (expected[0], expected[1] - 1, expected[2]):
            labels = labels.bool()
        else:
            raise ValueError("contacts must be [B,T,F] or [B,T-1,F]")
        feet_displacement = torch.diff(fk_pred[..., list(foot_indices), :], dim=1)
        if loss_protocol == "paper_v1":
            gated = torch.where(labels[..., None], feet_displacement.square(),
                                torch.zeros_like(feet_displacement))
            result["contact"] = masked_pose_sum_mean(gated, pair_mask)
        else:
            result["contact"] = masked_mean((feet_displacement * fps).square(),
                                            labels & pair_mask[..., None])
    return result
