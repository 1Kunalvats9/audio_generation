"""D2: conditional temporal U-Net, cosine diffusion and DDIM mel sampling.

Diffusion operates on NORMALISED (B, 80, T) mels. Each down/up stage masks
padding and normalises per frame, so no statistics cross clip boundaries.
The deterministic DDIM sampler uses the configured 50 steps by default.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.config import N_MELS
from src.dl.config import DECODER_KERNEL, DL, DROPOUT, HIDDEN
from src.dl.encoder import ConvBlock


def cosine_betas(steps: int) -> torch.Tensor:
    """Discretise alpha_bar(t)=cos²((t+.008)/1.008*pi/2), in float64."""
    if steps < 2:
        raise ValueError("diffusion needs at least two training steps")
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    alpha = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    alpha = alpha / alpha[0]
    return (1 - alpha[1:] / alpha[:-1]).clamp(min=1e-8, max=0.999).float()


class TimeEmbedding(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        if hidden < 4 or hidden % 2:
            raise ValueError("diffusion hidden size must be even and at least 4")
        frequencies = torch.exp(torch.arange(hidden // 2) * (-math.log(10000) / (hidden // 2 - 1)))
        self.register_buffer("frequencies", frequencies)
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, timesteps):
        phase = timesteps.float()[:, None] * self.frequencies[None, :]
        return self.proj(torch.cat([phase.sin(), phase.cos()], dim=-1))


class MelDenoiser(nn.Module):
    """One downsampling/upsampling U-Net with a skip connection and time FiLM bias."""
    def __init__(self, hidden=HIDDEN, kernel=DECODER_KERNEL, dropout=DROPOUT):
        super().__init__()
        self.input = nn.Linear(N_MELS + hidden, hidden)
        self.time = TimeEmbedding(hidden)
        self.down = ConvBlock(hidden, kernel, dropout)
        self.middle = ConvBlock(hidden, kernel, dropout)
        self.merge = nn.Linear(2 * hidden, hidden)
        self.up = ConvBlock(hidden, kernel, dropout)
        self.output = nn.Linear(hidden, N_MELS)

    def forward(self, noisy_mel, timesteps, frames, frame_mask):
        mask = frame_mask.unsqueeze(-1)
        mel = noisy_mel.transpose(1, 2).masked_fill(~mask, 0)
        frames = frames.masked_fill(~mask, 0)
        x = self.input(torch.cat([mel, frames], dim=-1))
        x = (x + self.time(timesteps)[:, None, :]).masked_fill(~mask, 0)
        skip = self.down(x, frame_mask)
        # Explicitly pad odd lengths: an appended padded frame must not change
        # the final pool's divisor relative to processing the clip alone.
        extra = skip.shape[1] % 2
        pooled = F.avg_pool1d(F.pad(skip.transpose(1, 2), (0, extra)), 2).transpose(1, 2)
        low_mask = F.max_pool1d(F.pad(frame_mask[:, None, :].float(), (0, extra)), 2)[:, 0].bool()
        low = self.middle(pooled, low_mask)
        up = low.repeat_interleave(2, dim=1)[:, :skip.shape[1]]
        x = self.merge(torch.cat([skip, up], dim=-1)).masked_fill(~mask, 0)
        x = self.up(x, frame_mask)
        return self.output(x).transpose(1, 2).masked_fill(~frame_mask[:, None, :], 0)


# Normalised mels occupy about -3.4 .. +3.4 (mean 0, std 1 by construction), so this
# bounds the predicted x0 generously without touching anything the data could produce.
CLEAN_CLAMP = 4.0

class MelDiffusion(nn.Module):
    def __init__(self, hidden=HIDDEN, train_steps=None, sample_steps=None):
        super().__init__()
        if DL["diffusion"]["schedule"] != "cosine":
            raise ValueError("only the agreed cosine schedule is supported")
        self.train_steps = DL["diffusion"]["train_steps"] if train_steps is None else train_steps
        self.sample_steps = DL["diffusion"]["sample_steps"] if sample_steps is None else sample_steps
        betas = cosine_betas(self.train_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0))
        self.denoiser = MelDenoiser(hidden)

    def q_sample(self, mel, timesteps, noise):
        alpha = self.alpha_bar[timesteps][:, None, None]
        return alpha.sqrt() * mel + (1 - alpha).sqrt() * noise

    def loss(self, mel, frames, frame_mask, *, timesteps=None, noise=None):
        if timesteps is None:
            timesteps = torch.randint(self.train_steps, (mel.shape[0],), device=mel.device)
        if noise is None:
            noise = torch.randn_like(mel)
        valid = frame_mask[:, None, :].expand_as(mel)
        mel = mel.masked_fill(~valid, 0)
        noise = noise.masked_fill(~valid, 0)
        noisy = self.q_sample(mel, timesteps, noise)
        pred = self.denoiser(noisy, timesteps, frames, frame_mask)
        error = pred.masked_select(valid) - noise.masked_select(valid)
        return error.square().sum() / valid.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, frames, frame_mask, steps=None, noise=None):
        """DDIM (eta=0): full train-time alpha schedule, shortened inference grid."""
        steps = self.sample_steps if steps is None else steps
        if not 1 <= steps <= self.train_steps:
            raise ValueError("sample steps must be between 1 and train_steps")
        shape = (frames.shape[0], N_MELS, frames.shape[1])
        if noise is None:
            noise = torch.randn(shape, device=frames.device, dtype=frames.dtype)
        if noise.shape != shape:
            raise ValueError("initial noise must match (B, 80, T)")
        valid = frame_mask[:, None, :]
        x = noise.masked_fill(~valid, 0)
        grid = torch.linspace(self.train_steps - 1, 0, steps, device=frames.device).long()
        was_training = self.training
        self.eval()
        try:
            for index, time in enumerate(grid):
                t = time.expand(frames.shape[0])
                predicted_noise = self.denoiser(x, t, frames, frame_mask)
                alpha = self.alpha_bar[time]
                clean = (x - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
                # alpha_bar[T-1] is 2.4e-09, so this division amplifies any error in
                # predicted_noise by ~20,000x on the first step and the sample diverges
                # (we measured mels at +/-2787 where the data spans +/-3.4). Clamping the
                # predicted clean signal to the data range is the standard DDIM guard.
                clean = clean.clamp(-CLEAN_CLAMP, CLEAN_CLAMP)
                previous = self.alpha_bar[grid[index + 1]] if index + 1 < steps else x.new_tensor(1.0)
                x = previous.sqrt() * clean + (1 - previous).sqrt() * predicted_noise
                x = x.masked_fill(~valid, 0)
        finally:
            self.train(was_training)
        return x
