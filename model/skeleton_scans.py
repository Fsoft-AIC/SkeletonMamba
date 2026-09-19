from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ssd_backend import make_mixer

JOINT_GROUPS = (
    (0, 3, 6, 9, 12, 15), (9, 14, 17, 19, 21, 23),
    (9, 13, 16, 18, 20, 22), (0, 2, 3, 5, 8, 11), (0, 1, 3, 4, 7, 10),
)
GROUP_PATHS = (
    (1, 2, 0, 3, 4), (2, 1, 0, 3, 4), (1, 2, 0, 4, 3), (2, 1, 0, 4, 3),
    (3, 4, 0, 1, 2), (4, 3, 0, 1, 2), (3, 4, 0, 2, 1), (4, 3, 0, 2, 1),
)


def validate_mask(mask: Tensor | None, batch: int, length: int,
                  device: torch.device, *, prefix: bool = True) -> Tensor:
    if batch < 1 or length < 1:
        raise ValueError("empty batches or zero-length sequences are not supported")
    if mask is None:
        return torch.ones(batch, length, dtype=torch.bool, device=device)
    if mask.shape != (batch, length) or mask.dtype != torch.bool or mask.device != device:
        raise ValueError("mask must be bool [batch,length] on the input device")
    if not mask.any(dim=1).all():
        raise ValueError("every sequence must have at least one valid frame")
    if prefix and (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError("motion/condition masks must describe a valid prefix")
    return mask


class HumanTokenizer(nn.Module):
    def __init__(self, num_joints: int = 24, groups=JOINT_GROUPS):
        super().__init__()
        indices = torch.as_tensor(groups, dtype=torch.long)
        if indices.ndim != 2 or indices.min() < 0 or indices.max() >= num_joints:
            raise ValueError("groups must be rectangular valid joint indices")
        counts = torch.bincount(indices.flatten(), minlength=num_joints)
        if (counts == 0).any():
            raise ValueError("tokenizer must cover every joint")
        self.num_joints = num_joints
        self.register_buffer("indices", indices)
        self.register_buffer("occurrences", counts)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2] != self.num_joints:
            raise ValueError("input joint dimension does not match tokenizer")
        return x[..., self.indices, :]

    def inverse(self, groups: Tensor) -> Tensor:
        if groups.shape[-3:-1] != self.indices.shape:
            raise ValueError("group tensor has incompatible group/joint dimensions")
        width = groups.shape[-1]
        flat = groups.flatten(-3, -2)
        result = groups.new_zeros(*groups.shape[:-3], self.num_joints, width)
        indices = self.indices.flatten().reshape(*([1] * (flat.ndim - 2)), -1, 1)
        result = result.scatter_add(-2, indices.expand_as(flat), flat)
        return result / self.occurrences.to(groups.dtype).unsqueeze(-1)


class GroupScan(nn.Module):
    def __init__(self, width: int, *, paths=GROUP_PATHS, backend="reference", **mixer_kwargs):
        super().__init__()
        paths = torch.as_tensor(paths, dtype=torch.long)
        if paths.ndim != 2 or not paths.shape[0] or not torch.equal(
            paths.sort(-1).values, torch.arange(5).expand_as(paths)
        ):
            raise ValueError("each group path must permute all five groups")
        self.register_buffer("paths", paths)
        self.register_buffer("inverse_paths", paths.argsort(-1))
        self.mixer = make_mixer(width, backend, **mixer_kwargs)

    def forward(self, groups: Tensor) -> Tensor:
        if groups.shape[-2] != 5:
            raise ValueError("group scan requires five anatomical groups")
        width, original_shape = groups.shape[-1], groups.shape
        groups = groups.reshape(-1, 5, width)
        batch, paths = groups.shape[0], self.paths.shape[0]
        # One sequence per frame; do not fold the path axis into the batch.
        sequenced = groups[:, self.paths, :].reshape(batch, paths * 5, width)
        scanned = self.mixer(sequenced).reshape(batch, paths, 5, width)
        inverses = self.inverse_paths[None, :, :, None].expand(batch, paths, 5, width)
        restored = scanned.gather(2, inverses).mean(1)
        return restored.reshape(original_shape)


class JointScan(nn.Module):
    def __init__(self, width: int, *, backend="reference", **mixer_kwargs):
        super().__init__()
        # Share mixer weights across anatomical groups.
        self.mixer = make_mixer(width, backend, **mixer_kwargs)

    def forward(self, groups: Tensor) -> Tensor:
        if groups.shape[-3:-1] != (5, 6):
            raise ValueError("joint scan requires [...,5,6,width]")
        return self.mixer(groups.reshape(-1, 6, groups.shape[-1])).reshape_as(groups)


class TemporalScan(nn.Module):
    def __init__(self, width: int, *, backend="reference", **mixer_kwargs):
        super().__init__()
        # Forward and backward scans have independent weights.
        self.forward_mixer = make_mixer(width, backend, **mixer_kwargs)
        self.backward_mixer = make_mixer(width, backend, **mixer_kwargs)
        self.output_projection = nn.Linear(width, width)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if x.ndim != 4:
            raise ValueError("temporal input must be [batch,time,joints,width]")
        batch, time, joints, width = x.shape
        mask = validate_mask(mask, batch, time, x.device)
        lengths = mask.sum(1)
        result = torch.zeros_like(x)
        for length in lengths.unique().tolist():
            rows = (lengths == length).nonzero(as_tuple=True)[0]
            selected = x.index_select(0, rows)[:, :length]
            sequences = selected.permute(0, 2, 1, 3).reshape(-1, length, width)
            forward = self.forward_mixer(sequences)
            backward = self.backward_mixer(sequences.flip(1)).flip(1)
            merged = self.output_projection(forward + backward)
            merged = merged.reshape(len(rows), joints, length, width).permute(0, 2, 1, 3)
            merged = F.pad(merged, (0, 0, 0, 0, 0, time - length))
            # Autocast projections can return BF16/FP16 while residuals remain
            # FP32. index_copy requires identical dtypes on both sides.
            result = result.index_copy(0, rows, merged.to(result.dtype))
        return result
