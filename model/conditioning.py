from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .skeleton_scans import validate_mask


def sinusoidal_embedding(positions: Tensor, width: int) -> Tensor:
    """Unbounded sinusoidal positions; integer diffusion indices are not rescaled."""
    if width < 2:
        raise ValueError("sinusoidal embedding width must be at least 2")
    half = width // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=positions.device, dtype=torch.float32) / max(half - 1, 1))
    angles = positions.float().unsqueeze(-1) * frequencies
    embedding = torch.cat((angles.sin(), angles.cos()), -1)
    return F.pad(embedding, (0, width - 2 * half))


class _FusionLayer(nn.Module):
    def __init__(self, width: int, num_heads: int, ff_size: int, dropout: float):
        super().__init__()
        self.audio_query = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
        self.video_query = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
        self.audio_norm = nn.LayerNorm(width)
        self.video_norm = nn.LayerNorm(width)
        self.audio_ff_norm = nn.LayerNorm(width)
        self.video_ff_norm = nn.LayerNorm(width)
        self.audio_ff = nn.Sequential(nn.Linear(width, ff_size), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(ff_size, width))
        self.video_ff = nn.Sequential(nn.Linear(width, ff_size), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(ff_size, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio: Tensor, video: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        an, vn = self.audio_norm(audio), self.video_norm(video)
        audio_update = self.audio_query(an, vn, vn, key_padding_mask=~mask, need_weights=False)[0]
        video_update = self.video_query(vn, an, an, key_padding_mask=~mask, need_weights=False)[0]
        audio = audio + self.dropout(audio_update)
        video = video + self.dropout(video_update)
        audio = audio + self.dropout(self.audio_ff(self.audio_ff_norm(audio)))
        video = video + self.dropout(self.video_ff(self.video_ff_norm(video)))
        return audio.masked_fill(~mask[..., None], 0), video.masked_fill(~mask[..., None], 0)


class MultiModalConditioner(nn.Module):
    def __init__(self, audio_dim: int, video_dim: int, context_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, num_heads: int = 4,
                 ff_size: int = 1024, dropout: float = 0.0,
                 fusion_architecture: str = "legacy_cross_attention_v1"):
        super().__init__()
        if hidden_dim % num_heads or num_layers < 1:
            raise ValueError("hidden width must divide heads and num_layers must be positive")
        if fusion_architecture not in {"legacy_cross_attention_v1", "paper_encoder_v1"}:
            raise ValueError(f"unknown fusion architecture: {fusion_architecture}")
        if fusion_architecture == "paper_encoder_v1" and hidden_dim != 2 * context_dim:
            raise ValueError("paper_encoder_v1 requires hidden_dim == 2 * context_dim")
        self.audio_dim, self.video_dim = audio_dim, video_dim
        self.hidden_dim = hidden_dim
        self.fusion_architecture = fusion_architecture
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.video_projection = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.GELU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        self.audio_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, ff_size, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.audio_alignment = nn.Linear(hidden_dim, context_dim)
        self.video_alignment = nn.Linear(hidden_dim, context_dim)
        if fusion_architecture == "legacy_cross_attention_v1":
            # Preserve checkpoint compatibility when architecture is omitted.
            self.fusion = nn.ModuleList([
                _FusionLayer(hidden_dim, num_heads, ff_size, dropout) for _ in range(num_layers)
            ])
            fusion_width = 2 * hidden_dim
        else:
            self.fusion = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_dim, num_heads, ff_size, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
                for _ in range(num_layers)
            ])
            fusion_width = hidden_dim
        self.context_projection = nn.Sequential(nn.Linear(fusion_width, context_dim),
                                               nn.LayerNorm(context_dim))

    def forward(self, audio: Tensor, video: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        if (audio.ndim != 3 or video.ndim != 3 or audio.shape[:2] != video.shape[:2]
                or audio.shape[-1] != self.audio_dim or video.shape[-1] != self.video_dim):
            raise ValueError("audio and video must be aligned [batch,time,configured_features]")
        batch, time = audio.shape[:2]
        mask = validate_mask(mask, batch, time, audio.device)
        # Exclude padding before projections as well as at attention boundaries.
        audio = self.audio_projection(audio.masked_fill(~mask[..., None], 0))
        video = self.video_projection(video.masked_fill(~mask[..., None], 0))
        positional = sinusoidal_embedding(torch.arange(time, device=audio.device),
                                          self.hidden_dim).to(audio.dtype)
        audio, video = audio + positional, video + positional
        for layer in self.audio_encoder:
            audio = layer(audio, src_key_padding_mask=~mask).masked_fill(~mask[..., None], 0)
        # Compute alignment before attention can mix the modalities.
        audio_embedding = F.normalize(self.audio_alignment(audio), dim=-1).masked_fill(~mask[..., None], 0)
        video_embedding = F.normalize(self.video_alignment(video), dim=-1).masked_fill(~mask[..., None], 0)
        if self.fusion_architecture == "legacy_cross_attention_v1":
            for layer in self.fusion:
                audio, video = layer(audio, video, mask)
            fused = torch.cat((audio, video), -1)
        else:
            # Fuse the same embeddings used by the alignment loss.
            fused = torch.cat((audio_embedding, video_embedding), -1)
            for layer in self.fusion:
                fused = layer(fused, src_key_padding_mask=~mask).masked_fill(~mask[..., None], 0)
        context = self.context_projection(fused).masked_fill(~mask[..., None], 0)
        return {"context": context, "context_mask": mask,
                "audio_embedding": audio_embedding, "video_embedding": video_embedding}
