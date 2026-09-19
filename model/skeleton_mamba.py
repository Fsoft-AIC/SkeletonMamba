"""Conditioned SkeletonMamba denoiser.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .conditioning import sinusoidal_embedding
from .skeleton_scans import GROUP_PATHS, GroupScan, HumanTokenizer, JointScan, TemporalScan, validate_mask


class DenseFiLM(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.projection = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * feature_dim))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.projection(condition).chunk(2, -1)
        shape = (x.shape[0], *([1] * (x.ndim - 2)), x.shape[-1])
        return x * (1 + scale.reshape(shape)) + shift.reshape(shape)


class SkeletonBlock(nn.Module):
    def __init__(self, width: int, context_dim: int, num_heads: int,
                 ff_size: int, dropout: float, paths, backend: str, mixer_kwargs: dict):
        super().__init__()
        self.tokenizer = HumanTokenizer()
        self.group_norm = nn.LayerNorm(6 * width)
        self.group_scan = GroupScan(6 * width, paths=paths, backend=backend, **mixer_kwargs)
        self.group_film = DenseFiLM(width, 6 * width)
        self.group_to_joints = nn.Linear(6 * width, 6 * width)
        self.joint_norm = nn.LayerNorm(width)
        self.joint_scan = JointScan(width, backend=backend, **mixer_kwargs)
        self.joint_film = DenseFiLM(width, width)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_scan = TemporalScan(width, backend=backend, **mixer_kwargs)
        self.temporal_film = DenseFiLM(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attention = nn.MultiheadAttention(width, num_heads, dropout=dropout,
                                                    batch_first=True, kdim=context_dim, vdim=context_dim)
        self.attention_film = DenseFiLM(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_size), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff_size, width))
        self.ff_film = DenseFiLM(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, condition: Tensor, context: Tensor,
                mask: Tensor, context_mask: Tensor) -> Tensor:
        batch, time, joints, width = x.shape
        grouped = self.tokenizer(x).flatten(-2)
        grouped = grouped + self.dropout(self.group_film(
            self.group_scan(self.group_norm(grouped)), condition))
        joint_tokens = self.group_to_joints(grouped).reshape(batch, time, 5, 6, width)
        joint_tokens = joint_tokens + self.dropout(self.joint_film(
            self.joint_scan(self.joint_norm(joint_tokens)), condition))
        x = self.tokenizer.inverse(joint_tokens).masked_fill(~mask[..., None, None], 0)
        x = x + self.dropout(self.temporal_film(
            self.temporal_scan(self.temporal_norm(x), mask), condition))
        queries = self.attention_norm(x).reshape(batch, time * joints, width)
        memory = self.context_norm(context)
        attended = self.cross_attention(queries, memory, memory,
                                        key_padding_mask=~context_mask, need_weights=False)[0]
        x = x + self.dropout(self.attention_film(attended.reshape_as(x), condition))
        x = x + self.dropout(self.ff_film(self.ff(self.ff_norm(x)), condition))
        return x.masked_fill(~mask[..., None, None], 0)


class SkeletonMamba(nn.Module):
    def __init__(self, input_dim: int = 9, embed_dim: int = 128, depth: int = 8,
                 num_joints: int = 24, d_context: int = 256,
                 backend: str = "reference", d_state: int = 64,
                 headdim: int = 64, expand: int = 2, ngroups: int = 1,
                 d_conv: int = 4, chunk_size: int = 64, num_heads: int = 4,
                 ff_size: int | None = None, dropout: float = 0.0,
                 group_paths=GROUP_PATHS, dt_min: float = 0.001, dt_max: float = 0.1,
                 dt_limit=(0.0, float("inf")), norm_before_gate: bool = False):
        super().__init__()
        if num_joints != 24:
            raise ValueError("this tokenizer requires the 24-joint SMPL skeleton")
        if embed_dim < 2 or embed_dim % num_heads or depth < 1:
            raise ValueError("embedding must divide attention heads; depth must be positive")
        self.input_dim, self.embed_dim, self.num_joints = input_dim, embed_dim, num_joints
        self.d_context = d_context
        mixer_kwargs = dict(d_state=d_state, headdim=headdim, expand=expand,
                            ngroups=ngroups, d_conv=d_conv, chunk_size=chunk_size,
                            dt_min=dt_min, dt_max=dt_max, dt_limit=dt_limit,
                            norm_before_gate=norm_before_gate)
        self.config = dict(input_dim=input_dim, embed_dim=embed_dim, depth=depth,
                           num_joints=num_joints, d_context=d_context, backend=backend,
                           num_heads=num_heads, ff_size=ff_size or 4 * embed_dim,
                           dropout=dropout, group_paths=[list(p) for p in group_paths],
                           **mixer_kwargs)
        self.reconstruction_choices = {
            "overlap_merge": "mean", "temporal_directions": "independent_sum_linear",
            "residual_norm": "pre_layernorm", "timestep_units": "integer_diffusion_index",
            "modulation": "dense_film_nonzero_init_with_pooled_context",
        }
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.joint_embedding = nn.Parameter(torch.randn(num_joints, embed_dim) * 0.02)
        self.time_embedding = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.SiLU(),
                                            nn.Linear(4 * embed_dim, embed_dim))
        self.context_pool_projection = nn.Sequential(nn.LayerNorm(d_context),
                                                     nn.Linear(d_context, embed_dim))
        self.blocks = nn.ModuleList([
            SkeletonBlock(embed_dim, d_context, num_heads, ff_size or 4 * embed_dim,
                          dropout, group_paths, backend, mixer_kwargs) for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, input_dim)
        # Preserve each mixer's dynamics initialization.

    def forward(self, x: Tensor, timesteps: Tensor, context: Tensor,
                mask: Tensor | None = None, context_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 4 or x.shape[2:] != (self.num_joints, self.input_dim):
            raise ValueError("motion must be [batch,time,24,input_dim]")
        batch, time = x.shape[:2]
        if (timesteps.shape != (batch,) or timesteps.dtype not in (torch.int32, torch.int64)
                or (timesteps < 0).any()):
            raise ValueError("timesteps must be nonnegative integer diffusion indices [batch]")
        if (context.ndim != 3 or context.shape[0] != batch
                or context.shape[-1] != self.d_context):
            raise ValueError("context must be [batch,context_time,d_context]")
        mask = validate_mask(mask, batch, time, x.device)
        context_mask = validate_mask(context_mask, batch, context.shape[1], x.device, prefix=False)
        context = context.masked_fill(~context_mask[..., None], 0)
        pooled = context.sum(1) / context_mask.sum(1, keepdim=True)
        condition = self.time_embedding(sinusoidal_embedding(timesteps, self.embed_dim).to(x.dtype))
        condition = condition + self.context_pool_projection(pooled)
        hidden = self.input_projection(x.masked_fill(~mask[..., None, None], 0))
        positions = sinusoidal_embedding(torch.arange(time, device=x.device), self.embed_dim).to(hidden.dtype)
        hidden = hidden + positions[None, :, None, :] + self.joint_embedding[None, None]
        hidden = hidden.masked_fill(~mask[..., None, None], 0)
        for block in self.blocks:
            hidden = block(hidden, condition, context, mask, context_mask)
        return self.output_projection(self.output_norm(hidden)).masked_fill(~mask[..., None, None], 0)
