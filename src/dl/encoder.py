"""Phoneme encoder and length regulator  (M2 — phase 2).

Two pieces, and the join between them is the heart of the FastSpeech 2 design:

    PhonemeEncoder   (B, P) phoneme ids      -> (B, P, H)  one vector per PHONEME
    LengthRegulator  (B, P, H) + durations   -> (B, T, H)  one vector per FRAME

The encoder reads the phoneme sequence; the length regulator stretches it onto
the time axis by repeating each phoneme's vector `duration` times. That is what
turns a 40-symbol sequence into a 300-frame one, and it is why durations are a
first-class input rather than something the decoder has to infer.

MASKING IS NOT OPTIONAL HERE. A convolution reaches across its kernel, so
without masking a padded slot bleeds into the real phonemes beside it and the
model's output depends on how long the *other* clips in the batch happened to
be. We zero padded positions before every convolution and again on the way out,
and `test_padding_does_not_change_real_positions` guards it.

Usage:
    from src.dl.encoder import build_encoder, LengthRegulator
    encoder = build_encoder()
    enc = encoder(batch["phone_ids"], batch["phone_mask"])          # (B, P, H)
    frames, frame_mask = LengthRegulator()(enc, batch["durations"]) # (B, T, H)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dl.config import (DROPOUT, ENCODER_KERNEL, ENCODER_LAYERS, HIDDEN,
                           PAD_ID)


def _zero_pad(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero every padded position. x (B, L, H), mask (B, L) with True = real."""
    return x * mask.unsqueeze(-1).to(x.dtype)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positions, so the encoder knows phoneme order.

    Not learned: the sequence lengths here are short and varied, and a fixed
    encoding extrapolates to lengths never seen in training.
    """

    def __init__(self, hidden: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / hidden))
        pe = torch.zeros(max_len, hidden)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        if L > self.pe.shape[0]:
            raise ValueError(f"sequence of {L} exceeds max_len={self.pe.shape[0]}")
        return x + self.pe[:L].unsqueeze(0)


class ConvBlock(nn.Module):
    """Pre-norm residual 1-D convolution: x + dropout(relu(conv(norm(x)))).

    Pre-norm (normalise before the conv rather than after) is the more stable of
    the two orderings to train without a warmup-heavy schedule.
    """

    def __init__(self, hidden: int, kernel: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1, "an odd kernel keeps the output length equal to the input"
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = _zero_pad(y, mask)                      # keep padding out of the kernel's reach
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        y = self.drop(F.relu(y))
        return _zero_pad(x + y, mask)


class PhonemeEncoder(nn.Module):
    """(B, P) phoneme ids -> (B, P, H) contextual phoneme representations."""

    def __init__(self, vocab_size: int, hidden: int = HIDDEN, layers: int = ENCODER_LAYERS,
                 kernel: int = ENCODER_KERNEL, dropout: float = DROPOUT):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=PAD_ID)
        self.pos = SinusoidalPositionalEncoding(hidden)
        self.blocks = nn.ModuleList(ConvBlock(hidden, kernel, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(hidden)
        nn.init.normal_(self.embed.weight, mean=0.0, std=hidden ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()

    def forward(self, phone_ids: torch.Tensor, phone_mask: torch.Tensor | None = None):
        if phone_mask is None:
            phone_mask = phone_ids != PAD_ID
        x = self.embed(phone_ids) * math.sqrt(self.hidden)
        x = self.pos(x)
        x = _zero_pad(x, phone_mask)
        for block in self.blocks:
            x = block(x, phone_mask)
        return _zero_pad(self.out_norm(x), phone_mask)


class LengthRegulator(nn.Module):
    """(B, P, H) + durations (B, P) -> (B, T, H), repeating each phoneme in time.

    Padded phonemes carry duration 0, so they contribute no frames and need no
    special handling. The operation has no parameters, but gradients flow back
    through the repeated vectors to the encoder.
    """

    def forward(self, x: torch.Tensor, durations: torch.Tensor, max_len: int | None = None):
        if durations.dtype.is_floating_point:
            durations = durations.round().clamp(min=0).long()
        if (durations < 0).any():
            raise ValueError("durations must be non-negative")

        expanded = [torch.repeat_interleave(x[b], durations[b], dim=0) for b in range(x.shape[0])]
        lengths = torch.tensor([e.shape[0] for e in expanded], device=x.device, dtype=torch.long)
        T = int(max_len if max_len is not None else lengths.max())

        out = x.new_zeros((x.shape[0], T, x.shape[2]))
        for b, e in enumerate(expanded):
            keep = min(e.shape[0], T)
            out[b, :keep] = e[:keep]

        frame_mask = torch.arange(T, device=x.device)[None, :] < lengths.clamp(max=T)[:, None]
        return out, frame_mask


def build_encoder(vocab_size: int | None = None) -> PhonemeEncoder:
    """Encoder sized from configs/dl.yaml and the frozen vocabulary."""
    if vocab_size is None:
        from src.dl.vocab import PhoneVocab
        vocab_size = PhoneVocab.load().size
    return PhonemeEncoder(vocab_size)


def _main() -> int:
    from src.dl.dataset import make_loader

    loader = make_loader("debug_200", batch_size=4, num_workers=0)
    batch = next(iter(loader))

    encoder = build_encoder().eval()
    with torch.no_grad():
        enc = encoder(batch["phone_ids"], batch["phone_mask"])
        frames, frame_mask = LengthRegulator()(enc, batch["durations"])

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"encoder: {n_params:,} parameters, hidden {encoder.hidden}")
    print(f"  phone_ids  {tuple(batch['phone_ids'].shape)} -> encoder {tuple(enc.shape)}")
    print(f"  + durations{tuple(batch['durations'].shape)} -> frames  {tuple(frames.shape)}")
    print(f"  frame_mask matches the dataset's: "
          f"{torch.equal(frame_mask, batch['frame_mask'])}")
    print(f"  padded phoneme slots are exactly zero: "
          f"{bool((enc[~batch['phone_mask']] == 0).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
