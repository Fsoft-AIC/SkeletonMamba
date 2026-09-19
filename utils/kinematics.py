"""Differentiable SMPL geometry, without licensed assets or compiled extensions.

6D rotations store the first two *rows* of a rotation matrix, matching
PyTorch3D. SMPL translation is not the pelvis position: pelvis = translation
plus the model's rest pelvis. All coordinates must share one declared frame.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


def _check_parents(parents, joints):
    parents = tuple(int(p) for p in parents)
    if len(parents) != joints or parents[0] != -1:
        raise ValueError("Parents must describe one root at joint zero")
    if any(p < 0 or p >= j for j, p in enumerate(parents[1:], 1)):
        raise ValueError("Parents must precede their children")
    return parents


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Rodrigues formula with analytic finite behavior at zero angle."""
    if axis_angle.shape[-1] != 3:
        raise ValueError("axis_angle must end in 3")
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(axis_angle.shape[:-1] + (3, 3))
    theta = torch.linalg.vector_norm(axis_angle, dim=-1)
    a = torch.sinc(theta / torch.pi)[..., None, None]
    b = (0.5 * torch.sinc(theta / (2 * torch.pi)).square())[..., None, None]
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return eye + a * skew + b * (skew @ skew)


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Stable real-first quaternion extraction, including rotations near pi."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must end in (3,3)")
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.reshape(*matrix.shape[:-2], 9).unbind(-1)
    squared = torch.stack((1+m00+m11+m22, 1+m00-m11-m22,
                           1-m00+m11-m22, 1-m00-m11+m22), -1)
    # Avoid sqrt's infinite derivative at zero in candidates not selected.
    positive = squared > 0
    q_abs = torch.where(positive, torch.sqrt(squared.clamp_min(1e-12)), torch.zeros_like(squared))
    candidates = torch.stack((
        torch.stack((q_abs[..., 0].square(), m21-m12, m02-m20, m10-m01), -1),
        torch.stack((m21-m12, q_abs[..., 1].square(), m10+m01, m02+m20), -1),
        torch.stack((m02-m20, m10+m01, q_abs[..., 2].square(), m12+m21), -1),
        torch.stack((m10-m01, m20+m02, m21+m12, q_abs[..., 3].square()), -1),
    ), -2) / (2 * q_abs.clamp_min(0.1)[..., :, None])
    choice = q_abs.argmax(-1)
    q = candidates.gather(-2, choice[..., None, None].expand(choice.shape+(1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    q = matrix_to_quaternion(matrix)
    half_angle = torch.atan2(torch.linalg.vector_norm(q[..., 1:], dim=-1), q[..., 0])
    scale = 0.5 * torch.sinc(half_angle / torch.pi)
    return q[..., 1:] / scale[..., None]


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must end in (3,3)")
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Gram-Schmidt, with deterministic fallback for degenerate predictions."""
    if d6.shape[-1] != 6:
        raise ValueError("6D rotations must end in 6")
    a1, a2 = d6[..., :3], d6[..., 3:]
    eps = 1e-8
    e1 = torch.zeros_like(a1)
    e1[..., 0] = 1
    b1 = torch.where(torch.linalg.vector_norm(a1, dim=-1, keepdim=True) > eps,
                     F.normalize(a1, dim=-1, eps=eps), e1)
    orthogonal = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # Choose the Cartesian axis least parallel to b1.
    axis = F.one_hot(b1.abs().argmin(-1), 3).to(d6)
    fallback = axis - (axis * b1).sum(-1, keepdim=True) * b1
    orthogonal = torch.where(torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True) > eps,
                             orthogonal, fallback)
    b2 = F.normalize(orthogonal, dim=-1, eps=eps)
    return torch.stack((b1, b2, torch.linalg.cross(b1, b2, dim=-1)), -2)


def rest_offsets(rest_joints: Tensor, parents=SMPL_PARENTS) -> Tensor:
    if rest_joints.ndim != 2 or rest_joints.shape[-1] != 3:
        raise ValueError("rest_joints must be [J,3]")
    parents = _check_parents(parents, len(rest_joints))
    return torch.stack([rest_joints[0]] + [rest_joints[j] - rest_joints[p]
                                         for j, p in enumerate(parents[1:], 1)])


def local_to_global_rotations(local_rotations: Tensor, parents=SMPL_PARENTS) -> Tensor:
    parents = _check_parents(parents, local_rotations.shape[-3])
    values = [local_rotations[..., 0, :, :]]
    for j, p in enumerate(parents[1:], 1):
        values.append(values[p] @ local_rotations[..., j, :, :])
    return torch.stack(values, -3)


def global_to_local_rotations(global_rotations: Tensor, parents=SMPL_PARENTS) -> Tensor:
    parents = _check_parents(parents, global_rotations.shape[-3])
    return torch.stack([global_rotations[..., 0, :, :]] + [
        global_rotations[..., p, :, :].transpose(-1, -2) @ global_rotations[..., j, :, :]
        for j, p in enumerate(parents[1:], 1)], -3)


def global_rotations_to_positions(global_rotations: Tensor, root_positions: Tensor,
                                 offsets: Tensor, parents=SMPL_PARENTS) -> Tensor:
    parents = _check_parents(parents, global_rotations.shape[-3])
    offsets = offsets.to(global_rotations)
    if offsets.shape != (len(parents), 3):
        raise ValueError("offsets must be [J,3]")
    positions = [root_positions]
    for j, p in enumerate(parents[1:], 1):
        positions.append(positions[p] + (global_rotations[..., p, :, :] @ offsets[j, :, None]).squeeze(-1))
    return torch.stack(positions, -2)


def forward_kinematics(local_rotations: Tensor, translation: Tensor, rest_joints: Tensor,
                       parents=SMPL_PARENTS) -> tuple[Tensor, Tensor]:
    offsets = rest_offsets(rest_joints.to(local_rotations), parents)
    global_rotations = local_to_global_rotations(local_rotations, parents)
    positions = global_rotations_to_positions(global_rotations, translation + offsets[0], offsets, parents)
    return global_rotations, positions
