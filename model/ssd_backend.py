from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _compute_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def ssd_scan(
    x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor,
    D: Tensor | None = None, *, dt_bias: Tensor | None = None,
    dt_softplus: bool = True, dt_limit: Sequence[float] = (0.0, float("inf")),
    initial_state: Tensor | None = None, return_final_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Differentiable causal recurrence; no CUDA/Triton dependency.

    x=[batch,length,heads,headdim], dt=[batch,length,heads], A=[heads],
    B/C=[batch,length,coefficient_groups,state_dim]. D is [heads] or
    [heads,headdim]; initial_state is [batch,heads,headdim,state_dim].
    A is the *continuous negative transition*, not its logarithm. Bias,
    softplus and optional limits are applied to dt exactly once.
    """
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("x must be floating [batch,length,heads,headdim]")
    batch, length, heads, headdim = x.shape
    if not length or dt.shape != (batch, length, heads) or A.shape != (heads,):
        raise ValueError("nonempty x, dt and A have incompatible shapes")
    if B.ndim != 4 or C.shape != B.shape or B.shape[:2] != (batch, length):
        raise ValueError("B and C must have equal [batch,length,groups,state] shapes")
    groups, state_dim = B.shape[2:]
    if not groups or heads % groups or not state_dim:
        raise ValueError("coefficient groups must divide SSM heads")
    if D is not None and D.shape not in ((heads,), (heads, headdim)):
        raise ValueError("D must be [heads] or [heads,headdim]")
    if dt_bias is not None and dt_bias.shape != (heads,):
        raise ValueError("dt_bias must be [heads]")
    if len(dt_limit) != 2 or not 0 <= dt_limit[0] <= dt_limit[1]:
        raise ValueError("invalid nonnegative dt_limit interval")
    dtype = _compute_dtype(x)
    delta = dt.to(dtype)
    if dt_bias is not None:
        delta = delta + dt_bias.to(dtype)
    if dt_softplus:
        delta = F.softplus(delta)
    delta = delta.clamp(min=dt_limit[0], max=dt_limit[1])
    xv, av = x.to(dtype), A.to(dtype)
    bv = B.to(dtype).repeat_interleave(heads // groups, dim=2)
    cv = C.to(dtype).repeat_interleave(heads // groups, dim=2)
    state_shape = (batch, heads, headdim, state_dim)
    if initial_state is None:
        state = torch.zeros(state_shape, dtype=dtype, device=x.device)
    elif initial_state.shape != state_shape:
        raise ValueError(f"initial_state must have shape {state_shape}")
    else:
        state = initial_state.to(dtype)
    skip = None if D is None else D.to(dtype).reshape(heads, -1)
    outputs = []
    for index in range(length):
        delta_i = delta[:, index]
        decay = torch.exp(delta_i * av)[..., None, None]
        injection = xv[:, index, :, :, None] * bv[:, index, :, None, :]
        state = decay * state + delta_i[..., None, None] * injection
        output = (state * cv[:, index, :, None, :]).sum(-1)
        if skip is not None:
            output = output + xv[:, index] * skip
        outputs.append(output)
    result = torch.stack(outputs, dim=1).to(x.dtype)
    return (result, state) if return_final_state else result


class GatedRMSNorm(nn.Module):
    """Mamba2 grouped RMSNorm, with configurable gate-before/after norm."""
    def __init__(self, width: int, ngroups: int = 1, eps: float = 1e-5,
                 norm_before_gate: bool = False):
        super().__init__()
        if width % ngroups:
            raise ValueError("RMSNorm groups must divide width")
        self.weight = nn.Parameter(torch.ones(width))
        self.ngroups, self.eps = ngroups, eps
        self.norm_before_gate = norm_before_gate

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        dtype = x.dtype
        values, gate = x.to(_compute_dtype(x)), gate.to(_compute_dtype(x))
        if not self.norm_before_gate:
            values = values * F.silu(gate)
        grouped = values.reshape(*values.shape[:-1], self.ngroups, -1)
        values = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True)
                                        + self.eps)).reshape_as(values)
        values = values * self.weight
        if self.norm_before_gate:
            values = values * F.silu(gate)
        return values.to(dtype)


class ReferenceMamba2Mixer(nn.Module):
    """Full-sequence recurrence without persistent caches.

    Parameter names match the supported subset of the official Mamba2 class,
    allowing strict state-dict transfer for numerical parity checks.
    """
    def __init__(self, d_model: int, d_state: int = 64, headdim: int = 64,
                 expand: int = 2, ngroups: int = 1, d_conv: int = 4,
                 dt_min: float = 0.001, dt_max: float = 0.1,
                 dt_init_floor: float = 1e-4,
                 dt_limit: Sequence[float] = (0.0, float("inf")),
                 A_init_range: Sequence[float] = (1.0, 16.0),
                 bias: bool = False, conv_bias: bool = True,
                 norm_before_gate: bool = False, chunk_size: int = 64):
        super().__init__()
        if min(d_model, d_state, headdim, expand, ngroups, d_conv, chunk_size) <= 0:
            raise ValueError("mixer dimensions must be positive")
        self.d_model, self.d_state, self.headdim = d_model, d_state, headdim
        self.d_inner, self.ngroups = expand * d_model, ngroups
        if self.d_inner % headdim or (self.d_inner // headdim) % ngroups:
            raise ValueError("expanded width must divide into heads and coefficient groups")
        if not 0 < dt_min <= dt_max or not 0 < A_init_range[0] <= A_init_range[1]:
            raise ValueError("invalid dynamics initialization range")
        self.nheads = self.d_inner // headdim
        self.d_conv, self.chunk_size = d_conv, chunk_size
        self.dt_limit = tuple(dt_limit)
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * ngroups * d_state
                                 + self.nheads, bias=bias)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, padding=d_conv - 1,
                               groups=conv_dim, bias=conv_bias)
        delta = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
                          + math.log(dt_min)).clamp_min(dt_init_floor)
        self.dt_bias = nn.Parameter(delta + torch.log(-torch.expm1(-delta)))
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(*A_init_range).log())
        self.D = nn.Parameter(torch.ones(self.nheads))
        for parameter in (self.dt_bias, self.A_log, self.D):
            parameter._no_weight_decay = True
            parameter._no_reinit = True
        self.norm = GatedRMSNorm(self.d_inner, ngroups, norm_before_gate=norm_before_gate)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, u: Tensor) -> Tensor:
        if u.ndim != 3 or u.shape[-1] != self.d_model or not u.shape[1]:
            raise ValueError("mixer input must be nonempty [batch,length,d_model]")
        batch, length, _ = u.shape
        z, xbc, dt = self.in_proj(u).split(
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], -1)
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[..., :length]).transpose(1, 2)
        x, b, c = xbc.split([self.d_inner, self.ngroups * self.d_state,
                            self.ngroups * self.d_state], -1)
        a = -torch.exp(self.A_log.to(_compute_dtype(u)))
        output = ssd_scan(x.reshape(batch, length, self.nheads, self.headdim), dt,
                          a, b.reshape(batch, length, self.ngroups, self.d_state),
                          c.reshape(batch, length, self.ngroups, self.d_state), self.D,
                          dt_bias=self.dt_bias, dt_limit=self.dt_limit)
        output = output.reshape(batch, length, self.d_inner)
        return self.out_proj(self.norm(output, z))


class OfficialMamba2Mixer(nn.Module):
    """Official CUDA kernels, with a layout adapter for unaligned projections.

    The combined convolution/scan requires channel-last strides divisible by
    eight. Unaligned joint and temporal projections use separate convolution
    and SSD kernels with a contiguous convolution input. Both paths expand D
    per channel to avoid a gradient reduction race in Mamba2 v2.2.4.
    """
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
        except (ImportError, OSError) as exc:
            raise RuntimeError("backend='mamba2' requires compatible official mamba-ssm, "
                               "Triton and causal-conv1d installations; use 'reference' "
                               "for CPU validation") from exc
        self.mixer = Mamba2(d_model=d_model, d_ssm=None, D_has_hdim=False,
                            rmsnorm=True, use_mem_eff_path=True, **kwargs)

    def forward(self, u: Tensor) -> Tensor:
        if not u.is_cuda:
            raise RuntimeError("backend='mamba2' requires CUDA; choose 'reference' for CPU")
        # Each diffusion step starts with a fresh SSM state.
        if self.mixer.in_proj.out_features % 8 == 0 and self.mixer.conv1d.in_channels % 8 == 0:
            return self._forward_combined_kernel(u)
        return self._forward_separate_kernels(u)

    def _expanded_skip(self):
        # Mamba2 v2.2.4's scalar-D backward overwrites partial channel-tile sums.
        # The per-channel branch lets ExpandBackward reduce them into the
        # original [heads] parameter, preserving values and state_dict shapes.
        return self.mixer.D[:, None].expand(-1, self.mixer.headdim).contiguous()

    def _forward_combined_kernel(self, u: Tensor) -> Tensor:
        """Upstream Mamba2's memory-efficient full-SSM/RMSNorm composition."""
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

        mixer = self.mixer
        return mamba_split_conv1d_scan_combined(
            mixer.in_proj(u), mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias,
            mixer.dt_bias, -torch.exp(mixer.A_log.float()), D=self._expanded_skip(),
            chunk_size=mixer.chunk_size, activation="silu", dt_limit=mixer.dt_limit,
            rmsnorm_weight=mixer.norm.weight, rmsnorm_eps=mixer.norm.eps,
            outproj_weight=mixer.out_proj.weight, outproj_bias=mixer.out_proj.bias,
            ngroups=mixer.ngroups, norm_before_gate=mixer.norm_before_gate,
        )

    def _forward_separate_kernels(self, u: Tensor) -> Tensor:
        """Compose the full-SSM/RMSNorm branch of upstream Mamba2 v2.2.4.

        Source: state-spaces/mamba, mamba_ssm/modules/mamba2.py (Tri Dao,
        Albert Gu). Parameters retain their original names in ``self.mixer``.
        """
        from causal_conv1d import causal_conv1d_fn
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        mixer = self.mixer
        batch, length, _ = u.shape
        coefficient_width = mixer.ngroups * mixer.d_state
        gate, xbc, dt = mixer.in_proj(u).split(
            [mixer.d_inner, mixer.d_inner + 2 * coefficient_width, mixer.nheads], -1)
        # Sequence-contiguous [B,C,L] avoids the channel-last stride restriction
        # in both causal-conv1d forward and backward, including short sequences.
        xbc = causal_conv1d_fn(
            xbc.transpose(1, 2).contiguous(), mixer.conv1d.weight.squeeze(1),
            bias=mixer.conv1d.bias, activation="silu",
        ).transpose(1, 2)
        x, b, c = xbc.split([mixer.d_inner, coefficient_width, coefficient_width], -1)
        output = mamba_chunk_scan_combined(
            x.reshape(batch, length, mixer.nheads, mixer.headdim), dt,
            -torch.exp(mixer.A_log.float()),
            b.reshape(batch, length, mixer.ngroups, mixer.d_state),
            c.reshape(batch, length, mixer.ngroups, mixer.d_state),
            chunk_size=mixer.chunk_size, D=self._expanded_skip(), dt_bias=mixer.dt_bias,
            dt_softplus=True, dt_limit=mixer.dt_limit,
        )
        return mixer.out_proj(mixer.norm(output.reshape(batch, length, mixer.d_inner), gate))


def make_mixer(d_model: int, backend: str = "reference", **kwargs) -> nn.Module:
    if backend == "reference":
        return ReferenceMamba2Mixer(d_model, **kwargs)
    if backend == "mamba2":
        return OfficialMamba2Mixer(d_model, **kwargs)
    raise ValueError(f"unknown SSD backend: {backend!r}")
