"""DDPM over explicit [B,T,24,9] motion; no ground-truth sampler inputs."""
import math

import torch
from torch import nn

from model.losses import validate_mask, masked_mean, geometry_losses, symmetric_alignment_loss


def extract(values, timesteps, shape):
    return values.gather(0, timesteps).reshape(-1, *([1] * (len(shape) - 1)))


def _full_precision(value):
    """Use at least float32 for DDPM math, preserving float64 inputs."""
    return value if value.dtype == torch.float64 else value.float()


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, objective="pred_x0", beta_schedule="cosine",
                 loss_type="l1", loss_weights=None, fps=30, alignment_temperature=0.1,
                 loss_protocol="legacy_xyz_v1"):
        super().__init__()
        if not isinstance(timesteps, int) or timesteps < 1:
            raise ValueError("timesteps must be a positive integer")
        aliases = {"x0": "pred_x0", "epsilon": "pred_noise", "eps": "pred_noise"}
        self.objective = aliases.get(objective, objective)
        if self.objective not in ("pred_x0", "pred_noise"):
            raise ValueError("objective must be pred_x0 or pred_noise")
        if loss_type not in ("l1", "l2"):
            raise ValueError("loss_type must be l1 or l2")
        if loss_protocol not in ("legacy_xyz_v1", "paper_v1"):
            raise ValueError("loss_protocol must be legacy_xyz_v1 or paper_v1")
        if not math.isfinite(fps) or not math.isfinite(alignment_temperature) or fps <= 0 or alignment_temperature <= 0:
            raise ValueError("fps and alignment_temperature must be positive")
        weights = dict(loss_weights or {})
        if set(weights) - {"position", "velocity", "contact", "consistency", "alignment"}:
            raise ValueError("unknown loss_weights keys")
        if any(not math.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError("loss weights must be finite and nonnegative")
        self.loss_weights = weights
        self.timesteps, self.num_timesteps = timesteps, timesteps
        self.loss_type, self.fps = loss_type, fps
        self.loss_protocol = loss_protocol
        self.alignment_temperature = alignment_temperature
        if beta_schedule == "cosine":
            s = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
            alpha_bar = torch.cos((s + 0.008) / 1.008 * math.pi / 2).square()
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        elif beta_schedule == "linear":
            betas = torch.linspace(0.0001 * 1000 / timesteps, 0.02 * 1000 / timesteps,
                                   timesteps, dtype=torch.float64).clamp(max=0.999)
        else:
            raise ValueError("beta_schedule must be cosine or linear")
        alphas = 1 - betas
        cumulative = torch.cumprod(alphas, 0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), cumulative[:-1]))
        buffers = {"betas": betas, "alphas_cumprod": cumulative,
                   "sqrt_alphas_cumprod": cumulative.sqrt(),
                   "sqrt_one_minus_alphas_cumprod": (1 - cumulative).sqrt(),
                   "posterior_variance": betas * (1 - previous) / (1 - cumulative),
                   "posterior_mean_coef1": betas * previous.sqrt() / (1 - cumulative),
                   "posterior_mean_coef2": (1 - previous) * alphas.sqrt() / (1 - cumulative)}
        for name, value in buffers.items():
            self.register_buffer(name, value.float())

    def q_sample(self, x_start, t, noise=None):
        with torch.autocast(device_type=x_start.device.type, enabled=False):
            x_start = _full_precision(x_start)
            noise = torch.randn_like(x_start) if noise is None else _full_precision(noise)
            return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                    extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            x_t, noise = _full_precision(x_t), _full_precision(noise)
            return ((x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * noise) /
                    extract(self.sqrt_alphas_cumprod, t, x_t.shape))

    def q_posterior(self, x_start, x_t, t):
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            x_start, x_t = _full_precision(x_start), _full_precision(x_t)
            mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                    extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
            return mean, extract(self.posterior_variance, t, x_t.shape)

    @staticmethod
    def _predict(model, x, t, conditions, mask):
        if "context" not in conditions:
            raise ValueError("conditions require cached context")
        output = model(x, t, conditions["context"], mask=mask,
                       context_mask=conditions.get("context_mask"))
        if output.shape != x.shape:
            raise ValueError("denoiser must return [B,T,24,9]")
        return output

    def training_losses(self, model, motion, conditions, mask=None, normalizer=None, *,
                        t=None, noise=None, contacts=None, offsets=None, parents=None):
        if motion.ndim != 4 or motion.shape[-2:] != (24, 9):
            raise ValueError("motion must be [B,T,24,9]")
        motion = _full_precision(motion)
        mask = validate_mask(mask, motion)
        clean = torch.where(mask[..., None, None], motion, torch.zeros_like(motion))
        t = (torch.randint(self.timesteps, (motion.shape[0],), device=motion.device)
             if t is None else t)
        if t.shape != (motion.shape[0],) or t.dtype != torch.long or (t < 0).any() or (t >= self.timesteps).any():
            raise ValueError("t must be integer diffusion indices [B]")
        noise = torch.randn_like(clean) if noise is None else noise
        if noise.shape != clean.shape:
            raise ValueError("noise must match motion shape")
        noise = torch.where(mask[..., None, None], noise, torch.zeros_like(noise))
        x_t = self.q_sample(clean, t, noise)
        # Only the network inherits the caller's mixed precision. Rotation/FK,
        # normalization, temporal reductions and contrastive logits remain FP32.
        output = self._predict(model, x_t, t, conditions, mask)
        with torch.autocast(device_type=motion.device.type, enabled=False):
            return self._training_loss_components(
                _full_precision(output), clean, _full_precision(noise), x_t, t,
                conditions, mask, normalizer, contacts, offsets, parents)

    def _training_loss_components(self, output, clean, noise, x_t, t, conditions,
                                  mask, normalizer, contacts, offsets, parents):
        target = clean if self.objective == "pred_x0" else noise
        difference = output - target
        reconstruction = masked_mean(difference.abs() if self.loss_type == "l1" else difference.square(), mask)
        pred_x0 = output if self.objective == "pred_x0" else self.predict_start_from_noise(x_t, t, output)
        pred_x0 = torch.where(mask[..., None, None], pred_x0, torch.zeros_like(pred_x0))
        components = geometry_losses(pred_x0, clean, mask, normalizer=normalizer,
                                     weights=self.loss_weights, offsets=offsets, parents=parents,
                                     contacts=contacts, fps=self.fps, loss_protocol=self.loss_protocol)
        components["alignment"] = reconstruction * 0
        if self.loss_weights.get("alignment", 0):
            audio = conditions.get("audio_embedding")
            video = conditions.get("video_embedding", conditions.get("vision_embedding"))
            if audio is None or video is None:
                raise ValueError("alignment requires audio_embedding and video_embedding")
            alignment_mask = conditions.get("alignment_mask", mask)
            if alignment_mask.shape != mask.shape or alignment_mask.dtype != torch.bool:
                raise ValueError("alignment_mask must be boolean [B,T]")
            alignment_mask = alignment_mask.to(mask.device) & mask
            components["alignment"] = symmetric_alignment_loss(audio, video, alignment_mask,
                                                                 self.alignment_temperature)
        loss = reconstruction + sum(self.loss_weights.get(name, 0) * value for name, value in components.items())
        return {"loss": loss, "reconstruction": reconstruction, **components, "pred_x0": pred_x0}

    @torch.no_grad()
    def sample(self, model, shape, conditions, mask=None, *, normalizer=None,
               head_guidance=None, generator=None, initial_noise=None):
        if len(shape) != 4 or tuple(shape[-2:]) != (24, 9) or min(shape) <= 0:
            raise ValueError("sample shape must be [B,T,24,9]")
        device = self.betas.device
        x = (torch.randn(tuple(shape), device=device, generator=generator)
             if initial_noise is None else initial_noise.detach().to(device).clone())
        x = _full_precision(x)
        if tuple(x.shape) != tuple(shape):
            raise ValueError("initial_noise must match requested shape")
        mask = validate_mask(mask, x)
        x = torch.where(mask[..., None, None], x, torch.zeros_like(x))
        states = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            for step in reversed(range(self.timesteps)):
                t = torch.full((shape[0],), step, dtype=torch.long, device=device)
                output = self._predict(model, x, t, conditions, mask)
                with torch.autocast(device_type=device.type, enabled=False):
                    output = _full_precision(output)
                    clean = output if self.objective == "pred_x0" else self.predict_start_from_noise(x, t, output)
                    protocol = None if head_guidance is None else getattr(head_guidance, 'protocol', None)
                    if head_guidance is not None and protocol not in ('posterior_log_v1', 'legacy_clean_frobenius_v1'):
                        raise ValueError('head guidance must declare a supported guidance protocol')
                    if protocol == 'legacy_clean_frobenius_v1':
                        clean = head_guidance(clean, mask=mask, normalizer=normalizer, timestep=t)
                    mean, variance = self.q_posterior(clean, x, t)
                    if protocol == 'posterior_log_v1':
                        mean = head_guidance.guide_posterior(mean, variance, mask=mask,
                                                             normalizer=normalizer, timestep=t)
                    noise = torch.randn(x.shape, dtype=x.dtype, device=device, generator=generator) if step else 0
                    x = mean + variance.sqrt() * noise
                    x = torch.where(mask[..., None, None], x, torch.zeros_like(x))
        finally:
            for module, training in states:
                module.training = training
        return x
