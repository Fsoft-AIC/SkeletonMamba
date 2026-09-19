"""Unaligned global-motion metrics with explicit physical units and masks.

Metrics use repository definitions; equivalence to unpublished KinPoly/paper
preprocessing is unverified. No floor or root alignment is performed.
Aggregate means are frame/contact weighted within the batch.
Undefined temporal/contact measurements return NaN and explicit counts.
"""
import math

import torch

from model.losses import validate_mask
from utils.kinematics import rotation_6d_to_matrix


def _mean(value, mask):
    mask = mask.to(device=value.device, dtype=torch.bool)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    if not mask.any():
        return value.new_tensor(float("nan"))
    return value[mask].mean()


@torch.no_grad()
def motion_metrics(prediction, target, mask=None, *, fps=30, position_unit="m",
                   head_index=15, foot_indices=(7, 8, 10, 11), contacts=None,
                   up_axis=2, floor_height=None, contact_height_threshold=0.05):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[-1] != 9:
        raise ValueError("prediction and target must share [B,T,J,9]")
    if not math.isfinite(fps) or fps <= 0 or up_axis not in (0, 1, 2) or not math.isfinite(contact_height_threshold) or contact_height_threshold < 0:
        raise ValueError("invalid FPS, up axis, or contact height threshold")
    if position_unit not in ("m", "mm"):
        raise ValueError("position_unit must be m or mm")
    mask = validate_mask(mask, prediction)
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("valid motion features must be finite")
    pred = torch.where(mask[..., None, None], prediction.float(), torch.zeros_like(prediction.float()))
    gt = torch.where(mask[..., None, None], target.float(), torch.zeros_like(target.float()))
    scale_to_m = 1.0 if position_unit == "m" else 0.001
    p, g = pred[..., :3] * scale_to_m, gt[..., :3] * scale_to_m
    distance = torch.linalg.vector_norm(p - g, dim=-1)
    pred_rot = rotation_6d_to_matrix(pred[..., head_index, 3:])
    gt_rot = rotation_6d_to_matrix(gt[..., head_index, 3:])
    # ||R1-R2||_F = 2*sqrt(2)*sin(theta/2). This is exactly zero for equal inputs.
    frobenius = torch.linalg.matrix_norm(pred_rot - gt_rot, ord="fro")
    angle = 2 * torch.asin((frobenius / (2 * 2**0.5)).clamp(0, 1))
    valid_accel = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    pred_accel = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) * fps**2
    gt_accel = (g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]) * fps**2
    result = {
        "mpjpe_mm": _mean(distance, mask) * 1000,
        "head_translation_mm": _mean(distance[..., head_index], mask) * 1000,
        "head_orientation_frobenius": _mean(frobenius, mask),
        "head_orientation_rad": _mean(angle, mask),
        "acceleration_error_mm_s2": _mean(torch.linalg.vector_norm(pred_accel - gt_accel, dim=-1), valid_accel) * 1000,
        "acceleration_pred_mm_s2": _mean(torch.linalg.vector_norm(pred_accel, dim=-1), valid_accel) * 1000,
        "acceleration_gt_mm_s2": _mean(torch.linalg.vector_norm(gt_accel, dim=-1), valid_accel) * 1000,
        "valid_frames": mask.sum(),
        "acceleration_frames": valid_accel.sum(),
    }
    pair_mask = mask[:, 1:] & mask[:, :-1]
    feet = p[..., list(foot_indices), :]
    if contacts is not None:
        labels = torch.as_tensor(contacts, device=prediction.device)
        if not torch.isfinite(labels).all() or ((labels != 0) & (labels != 1)).any():
            raise ValueError("contacts must be finite binary labels")
        expected = (*prediction.shape[:2], len(foot_indices))
        if labels.shape == expected:
            labels = labels[:, 1:].bool() & labels[:, :-1].bool()
        elif labels.shape == (expected[0], expected[1] - 1, expected[2]):
            labels = labels.bool()
        else:
            raise ValueError("contacts must be [B,T,F] or [B,T-1,F]")
    elif floor_height is not None:
        floor = torch.as_tensor(floor_height, dtype=feet.dtype, device=feet.device)
        if floor.numel() not in (1, prediction.shape[0]) or not torch.isfinite(floor).all():
            raise ValueError("floor_height must be finite metres, scalar or [B]")
        floor = floor.reshape(-1, 1, 1)
        grounded = (feet[..., up_axis] - floor).abs() <= contact_height_threshold
        labels = grounded[:, 1:] & grounded[:, :-1]
    else:
        labels = torch.zeros((*pair_mask.shape, len(foot_indices)), dtype=torch.bool, device=p.device)
    labels = labels & pair_mask[..., None]
    horizontal = [axis for axis in range(3) if axis != up_axis]
    foot_speed = torch.linalg.vector_norm(torch.diff(feet[..., horizontal], dim=1), dim=-1) * fps
    result["foot_skating_mm_s"] = _mean(foot_speed, labels) * 1000
    result["foot_contact_transitions"] = labels.sum()
    return result
