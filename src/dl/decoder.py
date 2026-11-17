"""D1: residual temporal decoder of normalised mel spectrograms (M4)."""

from __future__ import annotations

import torch
from torch import nn

from src.config import N_MELS
from src.dl.config import DECODER_KERNEL, DECODER_LAYERS, DROPOUT, HIDDEN
from src.dl.encoder import ConvBlock, SinusoidalPositionalEncoding


class MelDecoder(nn.Module):
    """Map length-regulated (B, T, H) features to (B, 80, T).

    Frame positions distinguish repeated vectors within a phoneme. LayerNorm
    and masking before every convolution keep clips independent of batch padding.
    """

    def __init__(self, hidden: int = HIDDEN, layers: int = DECODER_LAYERS,
                 kernel: int = DECODER_KERNEL, dropout: float = DROPOUT):
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(hidden)
        self.blocks = nn.ModuleList(ConvBlock(hidden, kernel, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, N_MELS)

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos(frames.masked_fill(~frame_mask.unsqueeze(-1), 0))
        x = x.masked_fill(~frame_mask.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, frame_mask)
        mel = self.proj(self.norm(x)).transpose(1, 2)
        return mel.masked_fill(~frame_mask.unsqueeze(1), 0)


def mel_loss(pred: torch.Tensor, target: torch.Tensor,
             frame_mask: torch.Tensor) -> torch.Tensor:
    """Mean L1 over real frames and mel bands, including unequal clip lengths."""
    if pred.shape != target.shape or frame_mask.shape != (pred.shape[0], pred.shape[2]):
        raise ValueError("expected matching (B, mel, T) tensors and a (B, T) mask")
    mask = frame_mask.unsqueeze(1).expand_as(pred)
    # Select before subtracting, so even NaNs in padding cannot poison the loss.
    error = (pred.masked_select(mask) - target.masked_select(mask)).abs()
    return error.sum() / mask.sum().clamp_min(1)
